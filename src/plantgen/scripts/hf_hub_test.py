import os
import sys
import yaml

from ..models.pipeline import PlantgenPipeline
from ..config.models import MMDiTConfig, ConvVAEConfig


def push_pipeline(config_path):
    config = yaml.safe_load(open(config_path, 'r'))

    config['vae']['ln_eps'] = float(config['vae']['ln_eps'])
    config['vae']['image_size'] = config['data']['image_size']

    vae_config = ConvVAEConfig(**config['vae'])
    model_config = MMDiTConfig(**config['model'])
    text_encoder = config['training']['text_encoder']

    vae_ckpt = config['training']['vae_ckpt_path']
    model_ckpt = f'{config["training"]["logdir"]}/model_{config["training"]["num_epochs"]}.pth'

    pipeline = PlantgenPipeline(
        model_config,
        vae_config,
        text_encoder
    )

    pipeline.load_ckpts(
        vae_ckpt,
        model_ckpt
    )

    img = pipeline.generate(
        ['Purple flower']
    )[0]

    os.makedirs('saved_images', exist_ok=True)
    img.save('saved_images/purple_flower.png')

    pipeline.push_to_hub(
        repo_id='groloch/Plantgen'
    )

def test_pipeline(config_path):
    pipeline = PlantgenPipeline.from_pretrained('groloch/Plantgen')

    img = pipeline.generate(
        ['Purple flower']
    )[0]

    os.makedirs('saved_images', exist_ok=True)
    img.save('saved_images/purple_flower_from_hub.png')


if __name__ == '__main__':
    config_path = sys.argv[1]
    
    # push_pipeline(config_path)
    test_pipeline(config_path)
