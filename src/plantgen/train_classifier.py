import torch
from tqdm import tqdm
from torchvision.models import (
    resnet18,
    resnet50,
    efficientnet_b0
)
import mlflow
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from .data import get_plantnet_dataloaders
from .metrics import MetricLogger
from .utils import model_parameters
from .config.data import PlantNetDataConfig
from .config.training import ClassifierTrainingConfig
from .config.tracking import MLflowConfig
from .models import MODEL_TYPE_DICT


def train_classifier(config):
    data_config = PlantNetDataConfig(**config['data'])
    training_config = ClassifierTrainingConfig(**config['training'])
    mlflow_config = MLflowConfig(**config['mlflow'])

    model_type = config['model_type']

    batch_size = data_config.batch_size

    train_dl, valid_dl = get_plantnet_dataloaders(data_config)

    model = MODEL_TYPE_DICT[model_type](num_classes=1081)
    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')

    model.to('cuda').to(torch.bfloat16)
    torch.compile(model, mode='max-autotune')

    mlflow.set_experiment(mlflow_config.experiment_name)
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.start_run()

    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=training_config.num_epochs * len(train_dl)
    )
    criterion = CrossEntropyLoss()

    logger = MetricLogger(use_mlflow=True)

    epochs = training_config.num_epochs
    for epoch in range(epochs):
        inputs: torch.Tensor
        targets: torch.Tensor
        loss: torch.Tensor
        outputs: torch.Tensor

        model.to(torch.bfloat16)
        model.train()

        for i, (inputs, targets) in (pbar := tqdm(enumerate(train_dl), desc=f'Epoch {epoch}', total=len(train_dl))):
            inputs = inputs.to('cuda').to(torch.bfloat16)
            targets = targets.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs).to(torch.float32)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, pred_top1 = outputs.max(1)
            _, pred_top5 = outputs.topk(5, 1, True, True)

            logger.update('loss', loss.item())
            logger.update('Top 1 accuracy', pred_top1.eq(targets).sum().item()/batch_size)
            logger.update('Top 5 accuracy', pred_top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()/batch_size)
            logger.update('lr', scheduler.get_last_lr()[0])
            if i % training_config.log_every == 0:
                if (i >= 100 or epoch > 0):
                    pbar.set_description(
                        f'Epoch {epoch} | {logger.log(step=epoch * len(train_dl) + i)}'
                    )
        model.eval()
        total = 0
        total_loss = 0.0
        total_top1 = 0
        total_top5 = 0

        with torch.inference_mode():
            for inputs, targets in tqdm(valid_dl, desc=f'Validation Epoch {epoch}', total=len(valid_dl)):
                inputs = inputs.to('cuda').to(torch.bfloat16)
                targets = targets.to('cuda')
                outputs = model(inputs).to(torch.float32)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, pred_top1 = outputs.max(1)
                total_top1 += pred_top1.eq(targets).sum().item()
                _, pred_top5 = outputs.topk(5, 1, True, True)
                total_top5 += pred_top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
                total += inputs.size(0)
        avg_loss = total_loss / total
        avg_top1 = total_top1 / total
        avg_top5 = total_top5 / total
        print(f"Validation Epoch {epoch}: loss={avg_loss:.4f}, Top-1 accuracy={avg_top1:.4f}, Top-5 accuracy={avg_top5:.4f}")

        if (epoch+1) % training_config.save_every == 0:
            torch.save(model.state_dict(), f'{training_config.logdir}/convvae_{epoch+1}.pth')
    mlflow.end_run()
