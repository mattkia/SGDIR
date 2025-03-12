import argparse
import logging
import os
import time
import torch


import numpy as np
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from models import FlowNet3D
from utils import SSIM3d
from utils import get_dataset
from utils import dice
from utils import compute_hd95
from utils import jacobian_determinant_3d


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='oasis',
                    help='JSON file for configuration')
# parse configs
args = parser.parse_args()

# configurations
torch.cuda.empty_cache()
config = Config(f'configs/{args.config}.yml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loadin the configs
batch_size = int(getattr(config, 'batch_size'))
res_levels = int(getattr(config, 'res_levels'))

ss_lr = float(getattr(config, 'ss_lr'))
ss_epochs = int(getattr(config, 'ss_epochs'))
ss_eval_interval = int(getattr(config, 'ss_eval_interval'))

loss_type = getattr(config, 'loss_type')
down_channels = getattr(config, 'down_channels') if hasattr(config, 'down_channels') else [32, 32, 32]
up_channels = getattr(config, 'up_channels') if hasattr(config, 'up_channels') else [32, 32, 32]
time_emb_dim = int(getattr(config, 'time_emb_dim')) if hasattr(config, 'time_emb_dim') else 64
decoder_only = getattr(config, 'decoder_only') if hasattr(config, 'decoder_only') else True

ckpt_path = getattr(config, 'checkpoints_path')

# creating required directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# defining the logger
logdir = getattr(config, 'logdir')
logging.basicConfig(filename=logdir, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# loading the dataset and dataloader
dataset = get_dataset(config=config, train=True)
loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

# loading the validation dataset
val_dataset = get_dataset(config=config, train=False, val=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


# initializing the network
network = FlowNet3D(down_channels=down_channels, 
                    up_channels=up_channels, 
                    time_emb_dim=time_emb_dim, 
                    loss_type=loss_type, 
                    decoder_only=decoder_only).to(device)

# setting up the optimizer
ss_optimizer = optim.Adam(network.parameters(), lr=ss_lr)

# setting loss factors
image_loss_coeff = float(getattr(config, 'image_loss_coeff'))
ss_loss_coeff = float(getattr(config, 'ss_loss_coeff'))


print('[*] Training on Continuous Semiring Regularization...')
logger.debug('Training with semiring regularization started...')

best_dice = 0.
ss_iters = tqdm(range(ss_epochs))
ss_iters.set_description('Loss: Inf')
ssim = SSIM3d()
for epoch in ss_iters:
    s_time = time.time()
    avg_loss = 0.
    avg_img_loss = 0.
    avg_ss_loss = 0.
    avg_f_dice = 0.
    avg_f_dice2 = 0.
    avg_f_det = 0.
    avg_tf_dice = 0.
    avg_tf_det = 0.
    avg_f_ssim = 0.
    avg_hd = 0.
    
    for sample in loader:
        if res_levels > 1:
            res = np.sort((0.95 - 0.05) * np.random.rand(res_levels - 1) + 0.05)
            res = np.concatenate((res, [1.0]))
        else:
            res = [1.0]
        
        for level in range(res_levels):
            I_, J_, xyz_, seg_I, seg_J = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device), sample[4].to(device)
            
            ss_optimizer.zero_grad()
            
            loss, ss_loss = network.loss_flow(I_, J_, xyz_, res[level])
            
            total_loss = image_loss_coeff * loss + ss_loss_coeff * ss_loss
            
            total_loss.backward()
            ss_optimizer.step()
            
            avg_img_loss += loss.item()
            avg_ss_loss += ss_loss.item()
            avg_loss += total_loss.item()
        
        with torch.no_grad():
            xyz = network(I_, J_, xyz_, torch.ones(1, device=device))
            xyzr = network(I_, J_, xyz_, -torch.ones(1, device=device))
            tf_dice, _ = dice(seg_I, seg_J, xyz, xyzr, structured=True)
            avg_tf_dice += tf_dice * 100
            avg_tf_det += jacobian_determinant_3d(xyz)
    
    avg_img_loss /= (res_levels * len(loader))
    avg_ss_loss /= (res_levels * len(loader))
    avg_loss /= (res_levels * len(loader))
    avg_tf_dice /= len(loader)
    avg_tf_det /= len(loader)

    for sample in val_loader:
        I_, J_, xyz_, seg_I, seg_J = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device), sample[4].to(device)
        
        with torch.no_grad():
            xyz = network(I_, J_, xyz_, torch.ones(1, device=device))
            xyzr = network(I_, J_, xyz_, -torch.ones(1, device=device))
            f_dice, _ = dice(seg_I, seg_J, xyz, xyzr, structured=True)
            f_dice2, _ = dice(seg_I, seg_J, xyz, xyzr, structured=False)
            avg_f_dice += f_dice * 100
            avg_f_dice2 += f_dice2 * 100
            avg_f_det += jacobian_determinant_3d(xyz)
            Jw = torch.nn.functional.grid_sample(J_, xyz, padding_mode='reflection', align_corners=True)
            warped_seg = torch.nn.functional.grid_sample(seg_J, xyz, padding_mode='reflection', mode='nearest', align_corners=True)
            avg_hd += compute_hd95(seg_I, warped_seg)
            avg_f_ssim += ssim(I_, Jw) * 100
        
    avg_f_dice /= len(val_loader)
    avg_f_dice2 /= len(val_loader)
    avg_f_det /= len(val_loader)
    avg_f_ssim /= len(val_loader)
    
    e_time = time.time()
    elapsed = (e_time - s_time) / 60.
    
    if epoch % ss_eval_interval == 0 or epoch == ss_epochs - 1:
        ss_iters.set_description(f'Loss: {avg_loss:.6f}')
        log = f'Epoch:{epoch}, Image Loss: {avg_img_loss:.6f}, Contraint Loss: {avg_ss_loss*ss_loss_coeff:.6f}, '
        log += f'Avg TF Dice: {avg_tf_dice:.2f}, Avg TF Det: {avg_tf_det:.4f}, '
        log += f'Avg F Dice: {avg_f_dice:.2f} ({avg_f_dice2:.2f}), Avg F Det: {avg_f_det:.4f}, Avf F SSIM: {avg_f_ssim:.2f}, Avg HD: {avg_hd:.2f}, Elapsed: {elapsed:.2f}m'
        
        if avg_f_dice2 > best_dice:
            best_dice = avg_f_dice2
            torch.save(network.state_dict(), ckpt_path)
            log += ' [CHCKPNT]'
            
        logger.debug(log)
        
        
