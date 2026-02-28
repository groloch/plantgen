import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet


class LogCoshLoss(nn.Module):
    """
    Smooth approximation of MAE (L1) loss
    """
    def __init__(self, a=2.0):
        super().__init__()
        self.a = a

    def forward(self, target, pred):
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(self.a * diff))) / self.a


class VAELoss(nn.Module):
    def __init__(self, steps_per_epoch: int):
        super(VAELoss, self).__init__()
        self.reconstruction_loss_fn = nn.L1Loss()

        self.perceptive_loss_fn = PerceptiveLoss(
            model_type='resnet18',
            layers_weights=[0.0625, 0.125, 0.25, 0.5]
        )

        self.kl_loss_weight = 5e-2
        self.perceptive_loss_weight = 1.0

    def forward(self, target, pred, mu, log_var):
        recon_loss = self.reconstruction_loss_fn(pred, target)
        perceptive_loss = self.perceptive_loss_fn(pred, target)
        total_loss = recon_loss + (self.perceptive_loss_weight * perceptive_loss)

        kl_loss = 0.5 * torch.mean(-1 - log_var + mu.pow(2) + log_var.exp())
        total_loss += (self.kl_loss_weight * kl_loss)
        return total_loss, recon_loss, perceptive_loss, kl_loss


class IAFLoss(nn.Module):
    """
    IAF KL-Divergence loss (https://arxiv.org/abs/1606.04934)
    """
    def __init__(self):
        super().__init__()

        self.reconstruction_loss_fn = nn.L1Loss()

        self.perceptive_loss_fn = PerceptiveLoss(
            model_type='resnet18',
            layers_weights=[0.0625, 0.125, 0.25, 0.5]
        )

        self.kl_loss_weight = 2e-3
        self.perceptive_loss_weight = 1.0

    def forward(self, target, pred, z, epsilon, *logvars):
        recon_loss = self.reconstruction_loss_fn(pred, target)
        perceptive_loss = self.perceptive_loss_fn(pred, target)
        total_loss = recon_loss + (self.perceptive_loss_weight * perceptive_loss)

        kl_loss = 0.5 * torch.mean(torch.square(z))
        kl_loss -= 0.5 * torch.mean(torch.square(epsilon))
        kl_loss -= torch.mean(sum(logvars))
        kl_loss = torch.abs(kl_loss)

        total_loss += (self.kl_loss_weight * kl_loss)
        return total_loss, recon_loss, perceptive_loss, kl_loss


class VAEFTLoss(VAELoss):
    def __init__(self, *args, **kwargs):
        super(VAEFTLoss, self).__init__(steps_per_epoch=1)
        self.perceptive_loss_fn = PerceptiveLoss(
            model_type='resnet18',
            weight_path='logs/classifier_training_r18/convvae_50.pth',
            layers_weights=[0.5, 0.25, 0.125, 0.0625]
        )

    def forward(self, target, pred, *args):
        recon_loss = self.perceptive_loss_fn(pred, target)

        return recon_loss, recon_loss, torch.tensor(0.0)


class PerceptiveLoss(nn.Module):
    """
    Perceptive loss as introduced by:
    https://arxiv.org/abs/1610.00291v2


    We used a ImageNet-1k pretrained ResNet-18 as the feature extractor, and the L1 distance for the loss.

    This is different from the original paper, where they used a VGG-19 model, but we found that
    using a more recent ResNet model gave better results.
    """
    def __init__(
            self,
            model_type: str,
            layers_weights: list[float]):
        super().__init__()

        self.model: ResNet

        self.model = resnet18(weights='IMAGENET1K_V1')

        self.model.eval()
        self.model.to('cuda', dtype=torch.bfloat16)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.compile(mode='max-autotune') # TODO guard this with a config parameter

        self.layers_weights = layers_weights

        self.dist_fn = nn.L1Loss()

    def forward(self, target, pred):
        x = self.model.conv1(pred)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        y = self.model.conv1(target)
        y = self.model.bn1(y)
        y = self.model.relu(y)
        y = self.model.maxpool(y)

        y1 = self.model.layer1(y)
        y2 = self.model.layer2(y1)
        y3 = self.model.layer3(y2)
        y4 = self.model.layer4(y3)

        loss_s1 = self.dist_fn(x1, y1)
        loss_s2 = self.dist_fn(x2, y2)
        loss_s3 = self.dist_fn(x3, y3)
        loss_s4 = self.dist_fn(x4, y4)

        loss = self.layers_weights[0] * loss_s1
        loss += self.layers_weights[1] * loss_s2
        loss += self.layers_weights[2] * loss_s3
        loss += self.layers_weights[3] * loss_s4

        return loss
