import os

import torch
import torchvision.transforms.v2 as transforms
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from ..models import ConvVAE
from ..utils import model_parameters, denormalize
from ..config.models import ConvVAEConfig
from ..config.training import VAETrainingConfig


def generate_and_vizualize(config: dict):
    print('Loading configs...')
    model_config = ConvVAEConfig(**config['model'])
    training_config = VAETrainingConfig(**config['training'])

    print('Logging dir:', training_config.logdir)
    save_dir = os.path.join(training_config.logdir, 'imgs')
    checkpoint = os.path.join(training_config.logdir, f'convvae_{training_config.num_epochs}.pth')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model...')
    model = ConvVAE(model_config)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # Compilation artefacts
    model.load_state_dict(ckpt)

    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')

    model.to(device)

    print('Generating images...')
    images = model.generate(batch_size=16, device=device)

    images = denormalize(images)
    
    images = images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    images = images.astype('uint8')

    for i, img in enumerate(images):
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f'img_{i}.png'))

def reconstruct_and_vizualize(config: dict):
    print('Loading configs...')
    model_config = ConvVAEConfig(**config['model'])
    training_config = VAETrainingConfig(**config['training'])

    print('Logging dir:', training_config.logdir)
    save_dir = os.path.join(training_config.logdir, 'imgs')
    checkpoint = os.path.join(training_config.logdir, f'convvae_{training_config.num_epochs}.pth')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model...')
    model = ConvVAE(model_config)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)

    ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # Compilation artefacts
    model.load_state_dict(ckpt)

    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = load_dataset('mikehemberger/plantnet300K')['test']

    indices = np.random.choice(len(dataset), size=16, replace=False)

    imgs = dataset.select(indices)['image']

    print(indices)
    transform =  transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    imgs = np.stack([transform(img) for img in imgs], axis=0)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=means, std=stds),
    ])
    
    # For grayscale VAE
    means = [0.5, 0.5, 0.5]
    stds = [0.25, 0.25, 0.25]
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=means, std=stds),
    ])

    inputs = torch.stack([transform(img) for img in imgs], dim=0).to(device)

    with torch.inference_mode():
        mu, logvar, _ = model.encode(inputs)
        z, _ = model.reparameterize(mu, logvar)

        mode_images = model.decode(mu)
        recon_images = model.decode(z)

        mode_images = denormalize(mode_images, means, stds)
        recon_images = denormalize(recon_images, means, stds)

    mode_images = mode_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    mode_images = mode_images.astype(np.uint8)

    recon_images = recon_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    recon_images = recon_images.astype(np.uint8)

    for i, (m_img, r_img) in enumerate(zip(mode_images, recon_images)):
        img = np.asarray(imgs[i])

        to_save = np.zeros((img.shape[0]*3, img.shape[1], 3), dtype=np.uint8)
        to_save[:img.shape[0], :, :] = img
        to_save[img.shape[0]:img.shape[0]*2, :, :] = r_img
        to_save[img.shape[0]*2:, :, :] = m_img

        to_save = Image.fromarray(to_save)
        to_save.save(os.path.join(save_dir, f'recon_img_{i}.png'))
        # to_save.show()

def vizualize_interpolation(config: dict):
    print('Loading configs...')
    model_config = ConvVAEConfig(**config['model'])
    training_config = VAETrainingConfig(**config['training'])

    print('Logging dir:', training_config.logdir)
    save_dir = os.path.join(training_config.logdir, 'imgs')
    checkpoint = os.path.join(training_config.logdir, f'convvae_{training_config.num_epochs}.pth')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model...')
    model = ConvVAE(model_config)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # rename all keys in state dict starting with _orig_mod. to remove this prefix
    ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)

    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')

    model.to(device)
    model.eval()

    dataset = load_dataset('mikehemberger/plantnet300K')['test']

    indices = np.random.choice(len(dataset), size=2, replace=False)

    indices = np.array([2109, 22393]) # 2 yellow flowers

    imgs = dataset.select(indices)['image']

    print(indices)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.to(device).unsqueeze(0))
    ])

    inputs = [transform(img) for img in imgs]

    inter_images = []

    with torch.inference_mode():
        latents = [model.encode(inp)[0] for inp in inputs]

        alphas = torch.linspace(0, 1, steps=100).unsqueeze(1).to(device)
        for alpha in tqdm(alphas):
            inter_z = alpha * latents[0] + (1 - alpha) * latents[1]
            inter_image = model.decode(inter_z)
            
            # Denormalize from ImageNet normalization to [0, 1]
            inter_image = denormalize(inter_image)

            inter_image = (inter_image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            inter_images.append(inter_image)

    pil_images = [Image.fromarray(img) for img in inter_images]
    pil_images[0].save(os.path.join(save_dir, 'interpolation.gif'), save_all=True, append_images=pil_images[1:], duration=100, loop=0)
