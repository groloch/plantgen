import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import mlflow

from .data import get_plantnet_dataloaders
from .models import ConvVAE
from .metrics import MetricLogger, VAELoss, VAEFTLoss, IAFLoss
from .utils import model_parameters
from .config.models import ConvVAEConfig
from .config.data import PlantNetDataConfig
from .config.training import VAETrainingConfig
from .config.tracking import MLflowConfig


def train_vae(config: dict):
    print('Loading configs...')
    model_config = ConvVAEConfig(**config['model'])
    data_config = PlantNetDataConfig(**config['data'])
    training_config = VAETrainingConfig(**config['training'])
    mlflow_config = MLflowConfig(**config['mlflow'])

    print('Logging dir:', training_config.logdir)

    print('Starting up tracking...')
    mlflow.set_experiment(mlflow_config.experiment_name)
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.start_run()

    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device('cuda')
    # device = torch.device('cpu')

    print('Initializing model...')
    model = ConvVAE(model_config)

    if training_config.ckpt_path is not None:
        print(f'Loading checkpoint...')
        ckpt = torch.load(training_config.ckpt_path, map_location='cpu', weights_only=True)

        # process all keys in state dict starting with _orig_mod. (compilation artefacts)
        ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)

        prepare_model(model, mode='train', config=training_config)

    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')
    if training_config.iaf:
        iaf_params, _ = model_parameters(model.iaf_model)
        print(f'IAF parameters: {iaf_params:,}')

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    print('Initializing datasets...')
    train_dataloader, valid_dataloader = get_plantnet_dataloaders(data_config)

    n_steps = len(train_dataloader) * training_config.num_epochs

    if training_config.ckpt_path is not None:
        vae_loss = VAEFTLoss(steps_per_epoch=len(train_dataloader))
    elif training_config.iaf == False:
        vae_loss = VAELoss(steps_per_epoch=len(train_dataloader))
    else:
        vae_loss = IAFLoss()

    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)

    train_logger = MetricLogger(use_mlflow=True)
    valid_logger = MetricLogger(use_mlflow=True)

    model.to(device)
    model.to(torch.bfloat16)

    # Compilation is currently disabled as it seems incompatible with IAF

    # print('Model compilation enabled')
    # model.compile(mode='max-autotune')

    print('Starting training...')
    for epoch in range(training_config.num_epochs):
        prepare_model(model, mode='train', config=training_config)
        train_one_epoch(
            epoch=epoch,
            model=model,
            loss_fn=vae_loss,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip=training_config.grad_clip,
            logger=train_logger,
            scheduler=scheduler,
            iaf=training_config.iaf,
            log_every=training_config.log_every
        )

        prepare_model(model, mode='valid', config=training_config)
        validate_one_epoch(
            epoch=epoch,
            model=model,
            loss_fn=vae_loss,
            valid_dataloader=valid_dataloader,
            device=device,
            logger=valid_logger,
            iaf=training_config.iaf,
            log_every=training_config.log_every
        )

        torch.cuda.empty_cache()

        if (epoch+1) % training_config.save_every == 0:
            torch.save(model.state_dict(), f'{training_config.logdir}/convvae_{epoch+1}.pth')

    mlflow.end_run()

def train_one_epoch(
        epoch: int,
        model: ConvVAE,
        loss_fn: VAELoss | IAFLoss,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        grad_clip: float,
        logger: MetricLogger,
        scheduler,
        iaf: bool,
        log_every: int):

    images: torch.Tensor

    for i, (images, _) in (
            pbar := tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch}', total=len(train_dataloader))
        ):
        optimizer.zero_grad()

        images = images.to(device, dtype=torch.bfloat16, non_blocking=True)

        if iaf:
            recon_images, z, eps, log_vars = model(images)
            loss, recon_loss, perceptive_loss, kl_loss = loss_fn(images, recon_images, z, eps, *log_vars)
        else:
            recon_images, mu, log_var = model(images)
            loss, recon_loss, perceptive_loss, kl_loss = loss_fn(images, recon_images, mu, log_var)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        logger.update('train_loss', loss.item())
        logger.update('recon_loss', recon_loss.item())
        logger.update('perceptive_loss', perceptive_loss.item())
        logger.update('kl_loss', kl_loss.item())

        logger.update('lr', scheduler.get_last_lr()[0])
        logger.update('kl_weight', loss_fn.kl_loss_weight)

        if (i >= 100 or epoch > 0) and i % log_every == 0:
            pbar.set_description(
                f'Epoch {epoch} | {logger.log(step=epoch * len(train_dataloader) + i)}'
            )


@torch.inference_mode()
def validate_one_epoch(
        epoch: int,
        model: ConvVAE,
        loss_fn: VAELoss,
        valid_dataloader: torch.utils.data.DataLoader,
        device: torch.device | str,
        logger: MetricLogger,
        iaf: bool,
        log_every: int):

    images: torch.Tensor

    for i, (images, _) in (
            pbar := tqdm(enumerate(valid_dataloader), desc=f'Validation {epoch}', total=len(valid_dataloader))
        ):

        images = images.to(device, dtype=torch.bfloat16, non_blocking=True)

        if iaf:
            recon_images, z, eps, log_vars = model(images)
            loss, recon_loss, perceptive_loss, kl_loss = loss_fn(images, recon_images, z, eps, *log_vars)
        else:
            recon_images, mu, log_var = model(images)
            loss, recon_loss, perceptive_loss, kl_loss = loss_fn(images, recon_images, mu, log_var)

        if i % log_every == 0:
            logger.update('valid_loss', loss.item())
            logger.update('valid_recon_loss', recon_loss.item())
            logger.update('valid_perceptive_loss', perceptive_loss.item())
            logger.update('valid_kl_loss', kl_loss.item())

            pbar.set_description(
                f'Validation {epoch} | {logger.log(step=epoch * len(valid_dataloader) + i)}'
            )

def prepare_model(
        model: ConvVAE,
        mode: str,
        config: VAETrainingConfig):

    if mode == 'train':
        model.train()
    else:
        model.eval()

    if config.ckpt_path is not None:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.mu_proj.parameters():
            param.requires_grad = False
        for param in model.logvar_proj.parameters():
            param.requires_grad = False
        for param in model.h_proj.parameters():
            param.requires_grad = False
