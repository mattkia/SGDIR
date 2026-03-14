"""Main file for evaluating 3D SGDIR
"""

import yaml
import torch
import pathlib
import argparse

from trainers import DiceTester
from trainers import TRETester


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        default='oasis',
                        help='JSON file for configuration')
    # parse configs
    args = parser.parse_args()

    # configurations
    torch.cuda.empty_cache()
    config_path = pathlib.Path('configs') / args.config
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)

    if config.get('data')['name'] in ['oasis', 'lpba40', 'ixi', 'candi', 'mindboggle', 'abdomen']:
        tester = DiceTester(config)
    elif config.get('data')['name'] in ['lungct']:
        tester = TRETester(config)

    tester.run()
