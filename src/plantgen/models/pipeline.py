import json
import os
import dataclasses
import tempfile

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
from PIL import Image

from ..models import ConvVAE, build_dit_model
from ..config.models import DiTConfig, ConvVAEConfig, CrossDITConfig, MMDiTConfig
from ..utils import denormalize


class PlantgenPipeline(nn.Module):
    def __init__(
            self,
            dit_config: DiTConfig,
            vae_config: ConvVAEConfig,
            encoder_path: str,
            device: torch.device='cuda'
        ):
        super().__init__()

        self.dit_config = dit_config
        self.vae_config = vae_config
        self.encoder_path = encoder_path

        self.vae = ConvVAE(vae_config)
        self.dit = build_dit_model(dit_config)
        self.text_encoder = AutoModel.from_pretrained(encoder_path)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path)

        self.latent_channels = dit_config.latent_dim
        self.latent_h = dit_config.latent_size
        self.latent_w = dit_config.latent_size

        self.device = device

        self.vae.to(self.device)
        self.dit.to(self.device)
        self.text_encoder.to(self.device)

    def load_ckpts(self, vae_ckpt_path: str, dit_ckpt_path: str):
        vae_state_dict = torch.load(vae_ckpt_path, weights_only=True)
        self.vae.load_state_dict(vae_state_dict)

        dit_state_dict = torch.load(dit_ckpt_path, weights_only=True)
        self.dit.load_state_dict(dit_state_dict)

    def forward(
            self,
            text_inputs: dict,
            batch_size: int,
            num_steps: int=30
        ):
        text_embed = self.text_encoder(**text_inputs).last_hidden_state

        latent_shape = (batch_size, self.latent_channels, self.latent_h, self.latent_w)

        z = torch.randn(latent_shape).to(self.device)

        for step in tqdm(range(num_steps), desc='Generating image'):
            t = torch.full((batch_size, 1, 1, 1), step / num_steps, device=self.device)
            v = self.dit(
                z,
                t,
                embeds=text_embed
            )
            z = z + v * (1.0 / num_steps)

        x = self.vae.decode(z)

        return x

    @torch.inference_mode()
    def generate(
            self,
            prompts: list[str],
            num_steps: int=30,
        ) -> list[Image.Image]:
        batch_size = len(prompts)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        x = self.forward(inputs, batch_size=batch_size, num_steps=num_steps)

        x = denormalize(x)

        x = x.cpu().permute(0, 2, 3, 1).numpy() * 255.0
        x = x.astype(np.uint8)

        images = [Image.fromarray(img) for img in x]
        return images
    
    def push_to_hub(self, repo_id: str, private: bool = False, token: str = None):
        """Push model weights and configs to the Hugging Face Hub."""
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, repo_type='model', exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'dit_config.json'), 'w') as f:
                json.dump(dataclasses.asdict(self.dit_config), f, indent=2)

            with open(os.path.join(tmpdir, 'vae_config.json'), 'w') as f:
                json.dump(dataclasses.asdict(self.vae_config), f, indent=2)

            with open(os.path.join(tmpdir, 'encoder_config.json'), 'w') as f:
                json.dump({'encoder_path': self.encoder_path}, f, indent=2)

            torch.save(self.vae.state_dict(), os.path.join(tmpdir, 'vae.pth'))
            torch.save(self.dit.state_dict(), os.path.join(tmpdir, 'dit.pth'))

            api.upload_folder(folder_path=tmpdir, repo_id=repo_id, repo_type='model')

    @classmethod
    def from_pretrained(
            cls,
            repo_id: str,
            device: torch.device = 'cuda',
            token: str = None
        ) -> 'PlantgenPipeline':
        """Load a PlantgenPipeline from the Hugging Face Hub."""
        from huggingface_hub import hf_hub_download

        def _download(filename):
            return hf_hub_download(repo_id=repo_id, filename=filename, token=token)

        with open(_download('dit_config.json')) as f:
            dit_config_dict = json.load(f)
        with open(_download('vae_config.json')) as f:
            vae_config_dict = json.load(f)
        with open(_download('encoder_config.json')) as f:
            encoder_path = json.load(f)['encoder_path']

        model_type = dit_config_dict.get('model_type')
        if model_type == 'crossdit':
            dit_config = CrossDITConfig(**dit_config_dict)
        elif model_type == 'mmdit':
            dit_config = MMDiTConfig(**dit_config_dict)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        vae_config = ConvVAEConfig(**vae_config_dict)

        pipeline = cls(
            dit_config=dit_config,
            vae_config=vae_config,
            encoder_path=encoder_path,
            device=device
        )

        vae_state_dict = torch.load(_download('vae.pth'), map_location='cpu', weights_only=True)
        pipeline.vae.load_state_dict(vae_state_dict)

        dit_state_dict = torch.load(_download('dit.pth'), map_location='cpu', weights_only=True)
        pipeline.dit.load_state_dict(dit_state_dict)

        return pipeline
