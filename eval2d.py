"""Main file for evaluating 2D SGDIR
"""

import yaml
import torch
import pathlib
import argparse

from trainers2d import DiceTester
from trainers2d import TRETester


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
    config_path = pathlib.Path('configs') / args.config
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)

    if config.get('data')['name'] in ['acdc']:
        tester = DiceTester(config)
    else:
        raise NotImplementedError()

    tester.run()
