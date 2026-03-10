import yaml
import torch
import argparse

from trainers2d import DiceTrainer
from trainers2d import TRETrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        default='acdc_unet',
                        help='JSON file for configuration')
    # parse configs
    args = parser.parse_args()

    # configurations
    torch.cuda.empty_cache()

    with open(f'configs/{args.config}.yml', 'r') as handle:
        config = yaml.safe_load(handle)

    if config.get('data')['name'] in ['acdc']:
        trainer = DiceTrainer(config)
    else:
        raise NotImplementedError()

    trainer.run()