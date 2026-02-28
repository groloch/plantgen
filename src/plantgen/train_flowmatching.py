import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import mlflow
from transformers import AutoModel, AutoTokenizer

from .data import get_plantnet_tti_dataloaders
from .models import ConvVAE, build_dit_model, DiTModel
from .metrics import MetricLogger
from .utils import model_parameters
from .config.models import ConvVAEConfig, DiTConfig
from .config.data import PlantNetTTIDataConfig
from .config.training import FlowMatchingTrainingConfig
from .config.tracking import MLflowConfig


def sample_timestep(
        distribution: str,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype
    ) -> torch.Tensor:
    if distribution == 'uniform':
        return torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype)
    elif distribution == 'logit_normal':
        u = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype)
        return torch.sigmoid(u)
    else:
        raise ValueError(f'Unknown timestep distribution: {distribution}')


def train_flowmatching(config: dict):
    print('Loading configs...')
    vae_config = ConvVAEConfig(**config['vae'])
    model_config = DiTConfig(**config['model'])
    data_config = PlantNetTTIDataConfig(**config['data'])
    training_config = FlowMatchingTrainingConfig(**config['training'])
    mlflow_config = MLflowConfig(**config['mlflow'])

    print('Logging dir:', training_config.logdir)

    if not training_config.profile:
        print('Starting up tracking...')
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.start_run()

    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device('cuda')

    print('Loading vae...')
    vae = ConvVAE(vae_config)
    vae_ckpt = torch.load(training_config.vae_ckpt_path, map_location='cpu', weights_only=True)
    vae_ckpt = {k.replace('_orig_mod.', ''): v for k, v in vae_ckpt.items()}
    vae.load_state_dict(vae_ckpt)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae_params = model_parameters(vae)[0]
    print(f'VAE parameters: {vae_params:,}')

    print('Loading text encoder...')
    tokenizer = AutoTokenizer.from_pretrained(training_config.text_encoder)
    text_encoder = AutoModel.from_pretrained(training_config.text_encoder)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder_params = model_parameters(text_encoder)[0]
    print(f'Text encoder parameters: {text_encoder_params:,}')

    print('Building model...')
    model = build_dit_model(model_config)
    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    print('Initializing datasets...')
    train_dataloader, valid_dataloader = get_plantnet_tti_dataloaders(data_config)

    n_steps = len(train_dataloader) * training_config.num_epochs

    loss_fn = torch.nn.MSELoss()

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=training_config.warmup_steps),
            CosineAnnealingLR(optimizer, T_max=n_steps - training_config.warmup_steps)
        ],
        milestones=[training_config.warmup_steps]
    )

    train_logger = MetricLogger(use_mlflow=not training_config.profile)
    valid_logger = MetricLogger(use_mlflow=not training_config.profile)

    vae.to(device)
    vae.to(torch.bfloat16)

    text_encoder.to(device)
    text_encoder.to(torch.bfloat16)

    model.to(device)
    model.to(torch.bfloat16)

    # print('Model compilation enabled')
    # model.compile(mode='max-autotune')

    if training_config.profile:
        profiling_run(
            model,
            vae,
            tokenizer,
            text_encoder,
            loss_fn,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            training_config,
            train_logger
        )
        return
    print('Starting training...')
    for epoch in range(training_config.num_epochs):
        model.train()
        train_one_epoch(
            epoch=epoch,
            model=model,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip=training_config.grad_clip,
            logger=train_logger,
            scheduler=scheduler,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            log_every=training_config.log_every,
            timestep_distribution=training_config.timestep_distribution
        )

        model.eval()
        validate_one_epoch()

        torch.cuda.empty_cache()

        if (epoch+1) % training_config.save_every == 0:
            torch.save(model.state_dict(), f'{training_config.logdir}/model_{epoch+1}.pth')

    mlflow.end_run()

def train_one_epoch(
        epoch: int,
        model: DiTModel,
        vae: ConvVAE,
        tokenizer: AutoTokenizer, # TODO better type hint
        text_encoder: AutoModel, # TODO better type hint
        loss_fn: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        grad_clip: float,
        logger: MetricLogger,
        scheduler,
        gradient_accumulation_steps: int,
        timestep_distribution: str,
        log_every: int,
        max_steps: int = None):

    images: torch.Tensor

    n_steps = len(train_dataloader)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    optimizer.zero_grad(set_to_none=True)
    training_step = 0
    total_steps = n_steps // gradient_accumulation_steps

    pbar = tqdm(total=total_steps, desc=f'Epoch {epoch}')

    loss_buffer = torch.zeros(
        gradient_accumulation_steps,
        device=device,
        dtype=torch.bfloat16
    )

    for i, (images, descriptions) in enumerate(train_dataloader):
        if max_steps is not None and i >= max_steps:
            break

        batch_size = images.size(0)

        images = images.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            latents, _, _ = vae.encode(images)

        with torch.no_grad():
            encodings = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True)
            input_ids = encodings['input_ids'].to(device)
            attn_mask = encodings['attention_mask'].to(device)
            embeds = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

        timestep = sample_timestep(
            distribution=timestep_distribution,
            batch_size=batch_size,
            device=device,
            dtype=latents.dtype
        )
        noise = torch.randn_like(latents)
        noised_latents = latents * timestep + noise * (1-timestep)
        targets = latents - noise

        outputs = model(noised_latents, timestep, embeds, attn_mask=attn_mask)

        loss = loss_fn(outputs, targets) / gradient_accumulation_steps
        loss.backward()

        loss_buffer[i % gradient_accumulation_steps] = loss.item() * gradient_accumulation_steps

        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == n_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            logger.update('train_loss', loss_buffer.mean().item())
            logger.update('lr', scheduler.get_last_lr()[0])
            training_step += 1
            pbar.update(1)

            loss_buffer.zero_()

            optimizer.zero_grad(set_to_none=True)

        if (training_step >= 100 or epoch > 0) and training_step % log_every == 0:
            pbar.set_description(
                f'Epoch {epoch} | {logger.log(step=epoch * total_steps + training_step)}'
            )

@torch.inference_mode()
def validate_one_epoch():
    pass

def profiling_run(
        model,
        vae,
        tokenizer,
        text_encoder,
        loss_fn,
        train_dataloader,
        optimizer,
        scheduler,
        device,
        training_config,
        train_logger):
    print('Warming up trainloop...')
    train_one_epoch(
        epoch=0,
        model=model,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        grad_clip=training_config.grad_clip,
        logger=train_logger,
        scheduler=scheduler,
        log_every=training_config.log_every,
        max_steps=10
    ) # Warmup
    print('Profiling trainloop...')
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
            activities=activities, with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        with record_function("trainloop"):
            train_one_epoch(
                epoch=0,
                model=model,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                loss_fn=loss_fn,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                grad_clip=training_config.grad_clip,
                logger=train_logger,
                scheduler=scheduler,
                log_every=training_config.log_every,
                max_steps=10
            )
    prof.export_chrome_trace(f"{training_config.logdir}/flowmatching_train_profile.json")
    prof.export_stacks(f"{training_config.logdir}/profiler_stacks.txt", "self_cuda_time_total")
    print(prof.key_averages(group_by_stack_n=10).table(sort_by="cpu_time_total", row_limit=20))


def train_flowmatching_precomp(config: dict):
    print('Loading configs...')
    model_config = DiTConfig(**config['model'])
    data_config = PlantNetTTIDataConfig(**config['data'])
    training_config = FlowMatchingTrainingConfig(**config['training'])
    mlflow_config = MLflowConfig(**config['mlflow'])

    print('Logging dir:', training_config.logdir)

    if not training_config.profile:
        print('Starting up tracking...')
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.start_run()

    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device('cuda')

    print('Loading text encoder...')
    tokenizer = AutoTokenizer.from_pretrained(training_config.text_encoder)
    text_encoder = AutoModel.from_pretrained(training_config.text_encoder)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder_params = model_parameters(text_encoder)[0]
    print(f'Text encoder parameters: {text_encoder_params:,}')

    print('Building model...')
    model = build_dit_model(model_config)
    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    print('Initializing datasets...')
    train_dataloader, valid_dataloader = get_plantnet_tti_dataloaders(data_config)

    n_steps = len(train_dataloader) * training_config.num_epochs

    loss_fn = torch.nn.MSELoss()

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=training_config.warmup_steps),
            CosineAnnealingLR(optimizer, T_max=n_steps - training_config.warmup_steps)
        ],
        milestones=[training_config.warmup_steps]
    )

    train_logger = MetricLogger(use_mlflow=not training_config.profile)
    valid_logger = MetricLogger(use_mlflow=not training_config.profile)

    text_encoder.to(device)
    text_encoder.to(torch.bfloat16)

    model.to(device)
    model.to(torch.bfloat16)

    # print('Model compilation enabled')
    # model.compile(mode='max-autotune')

    if training_config.profile:
        raise NotImplementedError("Profiling not implemented for precomputed latent training.")
    print('Starting training...')
    for epoch in range(training_config.num_epochs):
        model.train()
        train_one_epoch_precomp(
            epoch=epoch,
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip=training_config.grad_clip,
            logger=train_logger,
            scheduler=scheduler,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            log_every=training_config.log_every,
            timestep_distribution=training_config.timestep_distribution,
        )

        model.eval()
        validate_one_epoch()

        torch.cuda.empty_cache()

        if (epoch+1) % training_config.save_every == 0:
            torch.save(model.state_dict(), f'{training_config.logdir}/model_{epoch+1}.pth')

    mlflow.end_run()

def train_one_epoch_precomp(
        epoch: int,
        model: DiTModel,
        tokenizer: AutoTokenizer, # TODO better type hint
        text_encoder: AutoModel, # TODO better type hint
        loss_fn: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        grad_clip: float,
        logger: MetricLogger,
        scheduler,
        gradient_accumulation_steps: int,
        timestep_distribution: str,
        log_every: int,
        max_steps: int = None,
    ):

    n_steps = len(train_dataloader)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    optimizer.zero_grad(set_to_none=True)
    training_step = 0
    total_steps = n_steps // gradient_accumulation_steps

    pbar = tqdm(total=total_steps, desc=f'Epoch {epoch}')

    loss_buffer = torch.zeros(
        gradient_accumulation_steps,
        device=device,
        dtype=torch.bfloat16
    )

    for i, (latents, descriptions) in enumerate(train_dataloader):
        if max_steps is not None and i >= max_steps:
            break

        batch_size = latents.size(0)

        latents = latents.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            encodings = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True)
            input_ids = encodings['input_ids'].to(device)
            attn_mask = encodings['attention_mask'].to(device)
            embeds = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

        timestep = sample_timestep(
            distribution=timestep_distribution,
            batch_size=batch_size,
            device=device,
            dtype=latents.dtype
        )
        noise = torch.randn_like(latents)
        noised_latents = latents * timestep + noise * (1-timestep)
        targets = latents - noise

        outputs = model(noised_latents, timestep, embeds, attn_mask=attn_mask)

        loss = loss_fn(outputs, targets) / gradient_accumulation_steps
        loss.backward()

        loss_buffer[i % gradient_accumulation_steps] = loss.item() * gradient_accumulation_steps

        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == n_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            logger.update('train_loss', loss_buffer.mean().item())
            logger.update('lr', scheduler.get_last_lr()[0])
            training_step += 1
            pbar.update(1)

            loss_buffer.zero_()

            optimizer.zero_grad(set_to_none=True)

        if (training_step >= 100 or epoch > 0) and training_step % log_every == 0:
            pbar.set_description(
                f'Epoch {epoch} | {logger.log(step=epoch * total_steps + training_step)}'
            )

@torch.inference_mode()
def validate_one_epoch_precomp():
    pass


def train_flowmatching_interface(config: dict):
    if config['data']['precomputed_latents']:
        train_flowmatching_precomp(config)
    else:
        train_flowmatching(config)
