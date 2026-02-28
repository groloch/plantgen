import random

import json
import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..config.data import PlantNetDataConfig
from ..data import get_plantnet_dataloaders


def load_metadata():
    with open('data/class_idx_to_species_id.json', 'r') as f:
        idx_to_species = json.load(f)
    with open('data/plantnet300K_species_id_2_name.json', 'r') as f:
        species_to_name = json.load(f)
    with open('data/species_to_common_name.json', 'r') as f:
        species_to_common = json.load(f)

    mapping = {int(i): species_to_common[species_to_name[s]] for i, s in idx_to_species.items()}
    return mapping

def make_prompt(plant_species: str):
    description_prompts = [
        # 'Describe this image concisely.',
        'Provide a very short description of this image.',
    ]

    details_prompts = [
        f'Include plant species (the plant name is {plant_species}), and geometric details.',
        'Include color and lighting details.',
        'Include texture details.',
        f'Do not include background details (the plant name is {plant_species}). ',
        ''
    ]

    format_prompts = [
        'Do not use full sentences.',
        'Use short expressions separated by commas.',
        'Use a fantastic style.',
        'Use a realistic style.',
        'Use poetic language.',
        'Use technical botanical terminology.',
        'Use vivid adjectives.',
        'Use metaphorical descriptions.',
        'Use a scientific tone.',
        'Use impressionistic descriptions.',
        'Use minimal, sparse language.',
        'Use comparative descriptions (e.g., "like..." or "resembles...").',
        'Use emotive language.',
        'Use nature-inspired metaphors.',
        ''
    ]

    prompt = f'{random.choice(description_prompts)} {random.choice(details_prompts)} {random.choice(format_prompts)} Do not use emojis.'

    # prompt = f'{description_prompts[0]} {details_prompts[0]} {format_prompts[0]}'

    return prompt

def annotate_batch(images, plant_species: str, model: Qwen3VLForConditionalGeneration, processor: Qwen3VLProcessor):
    prompts = [make_prompt(plant_species=plant_species[k]) for k in range(images.shape[0])]

    messages = [
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'image': images[k],
                    },
                    {'type': 'text', 'text': prompts[k]},
                ],
            }
        ] for k in range(images.shape[0])
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        padding=True,
        padding_side='left',
        return_tensors='pt'
    )

    inputs = inputs.to(model.device)

    generation_config = model.generation_config
    generation_config.do_sample = True
    generation_config.top_p = 0.8
    generation_config.top_k = 20
    generation_config.temperature = 0.7
    generation_config.repetition_penalty = 1.0
    model.generation_config = generation_config

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def annotate_dataset():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-4B-Instruct', dtype=torch.bfloat16, device_map='cuda'
    )
    processor = Qwen3VLProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')

    species_mapping = load_metadata()

    data_config = PlantNetDataConfig(
        image_size=256,
        num_classes=1081,
        batch_size=64,
        num_workers=8,
        augmentation_mode='vlm'
    )
    train_loader, _ = get_plantnet_dataloaders(data_config, drop_last=False, shuffle_train=False)

    f = open('data/plantnet_captions.csv', 'w')
    text_buffer = []

    for i, (images, label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Annotating images'):
        plant_species = [species_mapping[l.item()] for l in label]
        outputs = annotate_batch(images, plant_species=plant_species, model=model, processor=processor)

        text_buffer.extend([repr(o) for o in outputs])

        # print(outputs)
        # plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
        # plt.title(outputs[0])
        # plt.show()

        # break

        if (i+1) % 50 == 0 or i == len(train_loader)-1:
            f.writelines('\n'.join(text_buffer) + '\n')
            text_buffer = []

    f.close()


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    annotate_dataset()
