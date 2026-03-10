import yaml
import torch
import argparse

from trainers2d import DiceTester
from trainers2d import TRETester


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
    tester = DiceTester(config)
else:
    raise NotImplementedError()

tester.run()