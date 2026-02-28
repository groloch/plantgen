import os
import sys

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

from ..models import ConvVAE
from ..utils import model_parameters, denormalize
from ..config.models import ConvVAEConfig
from ..config.data import PlantNetDataConfig
from ..config.training import VAETrainingConfig
from ..data import get_plantnet_dataloaders


def encode_dataset(config: dict, save_path: str):
    print('Loading configs...')
    model_config = ConvVAEConfig(**config['model'])
    data_config = PlantNetDataConfig(**config['data'])
    training_config = VAETrainingConfig(**config['training'])

    image_size = data_config.image_size
    latent_size = image_size // 8
    latent_dim = model_config.latent_dim
    batch_size = data_config.batch_size

    checkpoint = os.path.join(training_config.logdir, f'convvae_{training_config.num_epochs}.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model...')
    model = ConvVAE(model_config)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)

    ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # Compilation artefacts
    model.load_state_dict(ckpt)

    total_params, trainable_params = model_parameters(model)
    print(f'Model total parameters: {total_params:,}, trainable parameters: {trainable_params:,}')

    model.to(device, dtype=torch.float16)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    train_loader, _, = get_plantnet_dataloaders(
        data_config=data_config,
        drop_last=False,
        shuffle_train=False
    )

    n_shards = 25
    shard_size = len(train_loader) // n_shards
    print(f'{n_shards} shards, images per shard: {shard_size*batch_size}')
    shard_idx = 0
    current_shard = 0
    shard_buffer = np.zeros((shard_size, batch_size, latent_dim, latent_size, latent_size), dtype=np.float16)
    os.makedirs(save_path, exist_ok=True)

    for i, (inputs, _) in enumerate(tqdm(train_loader, desc='Encoding images')):
        inputs = inputs.to(device, dtype=torch.float16)

        mu: torch.Tensor

        mu, _, _ = model.encode(inputs)
        b = mu.shape[0]

        shard_buffer[shard_idx, :b, :, :, :] = mu.cpu().numpy()
        shard_idx += 1

        if shard_idx == shard_size or i == len(train_loader) - 1:
            shard_file = os.path.join(save_path, f'latents_shard_{current_shard:04d}_.npy')

            shard_buffer = shard_buffer.reshape(-1, latent_dim*latent_size*latent_size) # S*B, C*W*H
            if i == len(train_loader) - 1:
                image_count = (shard_idx-1) * data_config.batch_size + b
                shard_buffer = shard_buffer[:image_count, :]

            df = pd.DataFrame({'latents': list(shard_buffer)})
            df.to_parquet(shard_file.replace('.npy', '.parquet'))

            shard_idx = 0
            shard_buffer = np.zeros((shard_size, batch_size, latent_dim, latent_size, latent_size), dtype=np.float16)
            current_shard += 1
            print(f'Saved shard {current_shard} of {n_shards+1}')

    # df = pd.read_parquet(os.path.join(save_path, 'latents_shard_0000.parquet'))
    # latents = np.stack(df['latents'].to_numpy()).reshape(-1, 16, 64, 64)

    # latent = latents[1, :, :, :]
    # latent = torch.as_tensor(latent, dtype=torch.float16).unsqueeze(0).to(device)
    # with torch.inference_mode():
    #     recon = model.decode(latent)
    # recon = denormalize(recon)
    # recon = recon.cpu().permute(0, 2, 3, 1).numpy() * 255.0
    # recon = recon.astype(np.uint8)[0]
    # recon = Image.fromarray(recon)
    # recon.show()


if __name__ == '__main__':
    import yaml

    if len(sys.argv) < 3:
        print('Usage: python encode_dataset.py <config_path> <save_path> [image_size] [batch_size]')
        sys.exit(1)
    config_path = sys.argv[1]
    save_path = sys.argv[2]

    image_size = 128
    batch_size = 256
    if len(sys.argv) >= 4:
        image_size = int(sys.argv[3])
    if len(sys.argv) >= 5:
        batch_size = int(sys.argv[4])

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['model']['ln_eps'] = float(config['model']['ln_eps'])
    config['model']['image_size'] = config['data']['image_size']
    config['training']['iaf'] = config['model']['iaf']

    config['data']['image_size'] = image_size
    config['data']['batch_size'] = batch_size

    encode_dataset(config, save_path)
