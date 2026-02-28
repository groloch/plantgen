import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image

from ..config.data import PlantNetTTIDataConfig
from .plantnet_data import PlantNetTTIDataset, PlantNetPackedTTIDataset
from ..config.models import ConvVAEConfig
from ..models import ConvVAE
from ..utils import denormalize


if __name__ == "__main__":
    data_config = PlantNetTTIDataConfig(
        image_size=512,
        num_classes=1081,
        batch_size=1,
        num_workers=8,
        annotations_path="data/plantnet_captions.csv",
        similarity_threshold=0.,
        precomputed_latents=False
    )

    ldata_config = PlantNetTTIDataConfig(
        image_size=512,
        num_classes=1081,
        batch_size=1,
        num_workers=8,
        annotations_path="data/plantnet_captions.csv",
        similarity_threshold=0.,
        precomputed_latents=True,
        latents_path="data/plantnet_latents_512_r184/",
        latent_dim=16,
        latent_size=64
    )

    vae_config = ConvVAEConfig(
        in_channels=3,
        image_size=64,
        latent_dim=16,
        depths=[3, 5, 7, 3],
        dims=[48, 96, 192, 384],
        ln_eps=1e-6,
        iaf=False
    )
    vae = ConvVAE(vae_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = 'logs/vae_training_r184/convvae_3.pth'
    vae.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    vae.to(device, dtype=torch.float16)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    dataset = load_dataset('mikehemberger/plantnet300K')
    annotations = pd.read_csv(data_config.annotations_path, sep='\r')
    latent_dataset = load_dataset(ldata_config.latents_path)

    train_dataset = PlantNetTTIDataset(
        dataset['train'],
        annotations=annotations,
        image_size=data_config.image_size,
        threshold=data_config.similarity_threshold
    )

    packed_dataset = PlantNetPackedTTIDataset(
        latent_dataset['train'],
        annotations=annotations,
        latent_dim=ldata_config.latent_dim,
        latent_size=ldata_config.latent_size,
        threshold=ldata_config.similarity_threshold
    )
    assert len(train_dataset) == len(packed_dataset)

    idx = np.random.randint(0, len(train_dataset))
    print(idx)

    image, caption = train_dataset[idx]
    latent, lcaption = packed_dataset[idx]

    print(caption)
    print(lcaption)

    print(image.shape, latent.shape)
    image = image.unsqueeze(0)
    latent = latent.unsqueeze(0)

    with torch.inference_mode():
        latent = latent.to(device, dtype=torch.float16)
        reco_image = vae.decode(latent).to(dtype=torch.float32)
    
    image = denormalize(image)
    reco_image = denormalize(reco_image)

    image = image.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    image = image.astype(np.uint8)
    reco_image = reco_image.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255.0
    reco_image = reco_image.astype(np.uint8)

    W, H, C = image.shape
    _W, _H, _C = reco_image.shape
    assert W == _W and H == _H and C == _C

    display_image = np.zeros((W, H*2, C), dtype=np.uint8)
    display_image[:, :H, :] = image
    display_image[:, H:, :] = reco_image

    display_image = Image.fromarray(display_image)
    display_image.show()
