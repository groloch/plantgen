import os
import json

from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import gradio as gr
import torchvision.transforms.v2 as transforms

from ..models import ConvVAE, build_dit_model
from ..utils import model_parameters, denormalize
from ..config.training import FlowMatchingTrainingConfig
from ..config.models import ConvVAEConfig, DiTConfig


def run_app(config: dict):
    vae_config = ConvVAEConfig(**config['vae'])
    model_config = DiTConfig(**config['model'])
    training_config = FlowMatchingTrainingConfig(**config['training'])

    with open('data/class_idx_to_species_id.json', 'r') as f:
        class_idx_to_species_id = json.load(f)

    with open('data/plantnet300K_species_id_2_name.json', 'r') as f:
        species_id_to_name = json.load(f)

    with open('data/species_to_common_name.json', 'r') as f:
        species_to_common_name = json.load(f)

    class_choices = [('None', -1)]
    for idx in sorted(class_idx_to_species_id.keys(), key=int):
        species_id = class_idx_to_species_id[idx]
        scientific_name = species_id_to_name.get(species_id, f"Species {species_id}")
        common_name = species_to_common_name.get(scientific_name, "")
        display_name = f"{idx}: {scientific_name}" + (f" ({common_name})" if common_name else "")
        class_choices.append((display_name, int(idx)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading vae...')
    vae = ConvVAE(vae_config)
    vae_ckpt = torch.load(training_config.vae_ckpt_path, map_location='cpu', weights_only=True)
    vae_ckpt = {k.replace('_orig_mod.', ''): v for k, v in vae_ckpt.items()}
    vae.load_state_dict(vae_ckpt)
    vae.eval()
    vae.to(device)
    for param in vae.parameters():
        param.requires_grad = False
    vae_params = model_parameters(vae)[0]
    print(f'VAE parameters: {vae_params:,}')

    print('Loading text encoder...')
    tokenizer = AutoTokenizer.from_pretrained(training_config.text_encoder)
    text_encoder = AutoModel.from_pretrained(training_config.text_encoder)
    text_encoder.eval()
    text_encoder.to(device)
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder_params = model_parameters(text_encoder)[0]
    print(f'Text encoder parameters: {text_encoder_params:,}')

    print('Building model...')
    model = build_dit_model(model_config)
    model_ckpt = torch.load(f'{training_config.logdir}/model_{training_config.num_epochs}.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(model_ckpt)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    total_params = model_parameters(model)[0]
    print(f'Model parameters: {total_params:,}')

    @torch.inference_mode()
    def generate_images(
            prompt: str = None,
            num_images: int = 16,
            num_steps: int = 100
        ):

        batch_size = num_images
        latent_channels = model_config.latent_dim
        latent_h = model_config.latent_size
        latent_w = model_config.latent_size

        if prompt is not None:
            inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_embed = text_encoder(**inputs).last_hidden_state
            _, N, D = text_embed.shape
            text_embed = text_embed.expand(batch_size, N, D)
        else:
            text_embed = None

        latent_shape = (batch_size, latent_channels, latent_h, latent_w)

        z = torch.randn(latent_shape).to(device)

        for step in tqdm(range(num_steps), desc='Generating image'):
            t = torch.full((batch_size, 1, 1, 1), step / num_steps, device=device)
            v = model(
                z,
                t,
                embeds=text_embed
            )
            z = z + v * (1.0 / num_steps)

        x = vae.decode(z)
        x = denormalize(x)

        x = x.cpu().permute(0, 2, 3, 1).numpy() * 255.0
        x = x.astype(np.uint8)

        B, H, W, C = x.shape
        grid_size = int(np.sqrt(B))

        display_img = np.zeros(((H+10)*grid_size-10, (W+10)*grid_size-10, C), dtype=np.uint8)

        os.makedirs(f'{training_config.logdir}/img', exist_ok=True)
        for i in range(grid_size):
            for j in range(grid_size):
                img = Image.fromarray(x[i*grid_size + j, :, :, :])
                img.save(f'{training_config.logdir}/img/image_{i}_{j}.png')

                display_img[i*(H+10):i*(H+10)+H, j*(W+10):j*(W+10)+W, :] = x[i*grid_size + j, :, :, :]

        return Image.fromarray(display_img)

    with gr.Blocks(title="Plant image generation") as demo:

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter additional description...",
                    lines=2
                )
                with gr.Row():
                    num_images_input = gr.Slider(
                        minimum=4,
                        maximum=64,
                        value=16,
                        step=4,
                        label="Number of Images"
                    )
                    num_steps_input = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=10,
                        label="Generation Steps"
                    )
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Generated Images", type="pil")

        generate_btn.click(
            fn=generate_images,
            inputs=[prompt_input, num_images_input, num_steps_input],
            outputs=output_image
        )

    print("Starting Gradio app...")
    demo.launch()
