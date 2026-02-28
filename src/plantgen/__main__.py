import os
import sys
import yaml


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python main.py <config_file>\n')
        sys.exit(1)

    config_file = sys.argv[1]
    config = yaml.safe_load(open(config_file, 'r'))

    os.makedirs(config['training']['logdir'], exist_ok=True)
    if config['type'] in ('vae', 'classifier', 'flowmatching'):
        os.system(f'cp {config_file} {config['training']['logdir']}')
    
    # yaml loads floats in exponential notation as strings
    config['training']['learning_rate'] = float(config['training']['learning_rate'])

    if config['type'] in ('vae', 'vae_gen'):
        config['model']['ln_eps'] = float(config['model']['ln_eps'])
        config['model']['image_size'] = config['data']['image_size']
        config['training']['iaf'] = config['model']['iaf']
    elif config['type'] in ('flowmatching', 'flowmatching_demo'):
        config['vae']['ln_eps'] = float(config['vae']['ln_eps'])
        config['vae']['image_size'] = config['data']['image_size']
    
    match config['type']:
        case 'vae':
            from .train_vae import train_vae
            train_vae(config)
        case 'vae_gen':
            from .scripts.vae_gen import generate_and_vizualize, reconstruct_and_vizualize, vizualize_interpolation
            # generate_and_vizualize(config)
            reconstruct_and_vizualize(config)
            # vizualize_interpolation(config)
        case 'classifier':
            from .train_classifier import train_classifier
            train_classifier(config)
        case 'flowmatching':
            from .train_flowmatching import train_flowmatching_interface
            train_flowmatching_interface(config)
        case 'flowmatching_demo':
            from .scripts.demo import run_app
            run_app(config)
        case _:
            raise ValueError(f'Unknown run type: {config.get('type')}')
