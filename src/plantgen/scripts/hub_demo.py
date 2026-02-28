import torch
import gradio as gr
from PIL.Image import Resampling

from ..models.pipeline import PlantgenPipeline

TARGET_SIZE = (512, 512)


def run_app():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading pipeline from Hub...')
    pipeline = PlantgenPipeline.from_pretrained('groloch/Plantgen', device=device)
    pipeline.eval()
    print('Pipeline ready.')

    def generate(prompt: str, num_images: int, num_steps: int):
        prompts = [prompt] * num_images
        images = pipeline.generate(prompts, num_steps=num_steps)
        upscaled = [img.resize(TARGET_SIZE, resample=Resampling.BILINEAR) for img in images]
        return upscaled

    with gr.Blocks(title='Plantgen Demo') as demo:
        gr.Markdown('# Plantgen — Plant Image Generation\nGenerate plant images from a text prompt.')

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label='Prompt',
                    placeholder='Purple flower',
                    lines=2,
                )
                with gr.Row():
                    num_images_slider = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        label='Number of images',
                    )
                    num_steps_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        label='Diffusion steps',
                    )
                generate_btn = gr.Button('Generate', variant='primary')

            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label='Generated images (512x512)',
                    columns=4,
                    object_fit='contain',
                    height='auto',
                )

        generate_btn.click(
            fn=generate,
            inputs=[prompt_input, num_images_slider, num_steps_slider],
            outputs=gallery,
        )

    print('Starting Gradio app...')
    demo.launch()


if __name__ == '__main__':
    run_app()
