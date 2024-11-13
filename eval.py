import argparse
import logging
import os
import torch

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from models import FlowNet3D
from utils import SSIM3d
from utils import NCCLoss
from utils import dice
from utils import jacobian_determinant_3d
from utils import normalized_cross_correlation
from utils import save_images
from utils import save_warped
from utils import save_grid
from utils import animate_flow_3d
from utils import get_dataset
from utils import rolling_dice
from utils import save_flow_sequence
from utils import evaluate_semi_group_property


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='oasis',
                    help='JSON file for configuration')
# parse configs
args = parser.parse_args()

# configurations
torch.cuda.empty_cache()
config = Config(f'configs/{args.config}.yml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = torch.load(getattr(config, 'checkpoints_path'))
save_evals = getattr(config, 'save_evals')
save_evals_path = getattr(config, 'save_evals_path')
batch_size = int(getattr(config, 'batch_size'))
down_factor = int(getattr(config, 'down_factor'))

down_channels = getattr(config, 'down_channels') if hasattr(config, 'down_channels') else [64, 128, 256, 512]
up_channels = getattr(config, 'up_channels') if hasattr(config, 'up_channels') else [512, 256, 128, 64]
time_emb_dim = int(getattr(config, 'time_emb_dim')) if hasattr(config, 'time_emb_dim') else 64
decoder_only = getattr(config, 'decoder_only') if hasattr(config, 'decoder_only') else True

# creating required directories
os.makedirs(save_evals_path, exist_ok=True)

# defining the logger
logging.basicConfig(filename=getattr(config, 'eval_logdir'), format='%(asctime)s %(message)s', filemode='w')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# defining the evaluation dataset
dataset = get_dataset(config=config, train=False)
loader = DataLoader(dataset, batch_size=batch_size)


# loading the model
network = FlowNet3D(down_channels=down_channels, 
                    up_channels=up_channels, 
                    time_emb_dim=time_emb_dim, 
                    decoder_only=decoder_only).to(device)
network.load_state_dict(weights)

ssim = SSIM3d()
ncc = NCCLoss(win=11)
# evaluating the model
with torch.no_grad():
    print('[*] Evaluation started...')
    logger.debug(f'Evaluation started on {len(dataset)} samples...')
    avg_neg_jac_dets_forward = []
    avg_neg_jac_dets_reverse = []
    avg_ncc_forward = []
    avg_ncc_reverse = []
    avg_local_ncc_forward = []
    avg_local_ncc_reverse = []
    avg_init_ncc = 0.
    avg_init_local_ncc = 0.
    avg_difference_mean_forward = 0.
    avg_difference_mean_reverse = 0.
    avg_difference_std_forward = 0.
    avg_difference_std_reverse = 0.
    avg_forward_dice = []
    avg_reverse_dice = []
    avg_forward_dice2 = []
    avg_reverse_dice2 = []
    avg_forward_ssim = []
    avg_reverse_ssim = []
    
    for batch_id, sample in enumerate(tqdm(loader)):
        I_, J_, xyz_, seg_I_, seg_J_ = sample
        
        I_ = I_.to(device)
        J_ = J_.to(device)
        xyz_ = xyz_.to(device)
        seg_I_ = seg_I_.to(device)
        seg_J_ = seg_J_.to(device)
        
        t = torch.ones(1, dtype=torch.float32).to(device)
        xyz = network(I_, J_, xyz_, t)
        xyzr = network(I_, J_, xyz_, -t)
        
        Jw = F.grid_sample(J_, xyz, padding_mode='reflection', align_corners=True)
        Iw = F.grid_sample(I_, xyzr, padding_mode='reflection', align_corners=True)
        seg_Jw = F.grid_sample(seg_J_, xyz, padding_mode='reflection', align_corners=True, mode='nearest')
        
        neg_jac_det_forward = jacobian_determinant_3d(xyz).item()
        neg_jac_det_reverse = jacobian_determinant_3d(xyzr).item()
        
        ncc_forward = normalized_cross_correlation(I_, Jw).item()
        ncc_reverse = normalized_cross_correlation(J_, Iw).item()
        init_ncc = normalized_cross_correlation(I_, J_).item()
        init_local_ncc = ncc(I_, J_).item()
        local_ncc_forward = ncc(I_, Jw).item()
        local_ncc_reverse = ncc(J_, Iw).item()
        
        difference_forward = (Jw - I_).abs()
        difference_reverse = (Iw - J_).abs()
        
        forward_dice, reverse_dice = dice(seg_I_, seg_J_, xyz, xyzr, structured=False)
        forward_dice2, reverse_dice2 = dice(seg_I_, seg_J_, xyz, xyzr, structured=True)
        
        avg_forward_dice.append(forward_dice.item())
        avg_reverse_dice.append(reverse_dice.item())
        avg_forward_dice2.append(forward_dice2.item())
        avg_reverse_dice2.append(reverse_dice2.item())
        
        avg_forward_ssim.append(ssim(I_, Jw).item())
        avg_reverse_ssim.append(ssim(J_, Iw).item())
        
        avg_difference_mean_forward += difference_forward.mean().item()
        avg_difference_mean_reverse += difference_reverse.mean().item()
        
        avg_difference_std_forward += difference_forward.std().item()
        avg_difference_std_reverse += difference_reverse.std().item()
        
        avg_neg_jac_dets_forward.append(neg_jac_det_forward)
        avg_neg_jac_dets_reverse.append(neg_jac_det_reverse)
        avg_ncc_forward.append(ncc_forward)
        avg_ncc_reverse.append(ncc_reverse)
        avg_local_ncc_forward.append(local_ncc_forward)
        avg_local_ncc_reverse.append(local_ncc_reverse)
        avg_init_ncc += init_ncc
        avg_init_local_ncc += init_local_ncc
        
        if save_evals:
            mid_frame_index = I_.size(3) // 2
            rolling_dice(network, I_, J_, seg_I_, seg_J_, xyz_, getattr(config, 'save_evals_path'), num_frames=50)
            save_images(I_[:, :, :, mid_frame_index, :], J_[:, :, :, mid_frame_index, :], 
                        Iw[:, :, :, mid_frame_index, :], Jw[:, :, :, mid_frame_index, :], 
                        save_evals_path, show_diff=False, prefix=str(batch_id))
            save_warped(Jw[:, :, :, mid_frame_index, :], seg_Jw[:, :, :, mid_frame_index, :], save_path=save_evals_path, prefix=str(batch_id))
            
            xy = torch.cat([xyz[:, :, mid_frame_index, :, 0].unsqueeze(-1), xyz[:, :, mid_frame_index, :, 2].unsqueeze(-1)], dim=-1)
            xyr = torch.cat([xyzr[:, :, mid_frame_index, :, 0].unsqueeze(-1), xyzr[:, :, mid_frame_index, :, 2].unsqueeze(-1)], dim=-1)
            xy = xy[:, ::down_factor, ::down_factor, :]
            xyr = xyr[:, ::down_factor, ::down_factor, :]
            save_grid(xy, xyr, save_path=save_evals_path, prefix=str(batch_id))
            save_flow_sequence(network, I_[0].unsqueeze(0), J_[0].unsqueeze(0), xyz_[0].unsqueeze(0),
                               getattr(config, 'save_evals_path'), num_frames=7)
            animate_flow_3d(network, I_[0].unsqueeze(0), J_[0].unsqueeze(0), xyz_[0].unsqueeze(0), 
                            save_path=getattr(config, 'save_evals_path'), down_factor=down_factor, num_frames=50, prefix=f'batch_{batch_id}')
    
    
    avg_init_ncc /= len(loader)
    avg_init_local_ncc /= len(loader)
    avg_difference_mean_forward /= len(loader)
    avg_difference_mean_reverse /= len(loader)
    avg_difference_std_forward /= len(loader)
    avg_difference_std_reverse /= len(loader)
    
    logger.debug('Evaluation Finished')
    logger.debug(f'Forward Absolute Difference: {avg_difference_mean_forward:.6f} +- {avg_difference_std_forward:.6f}')
    logger.debug(f'Average Forward NCC: {np.mean(avg_ncc_forward) * 100:.3f} +- {np.std(avg_ncc_forward) * 100:.3f} (Original NCC: {avg_init_ncc * 100:.3f})')
    logger.debug(f'Average Forward Local NCC: {np.mean(avg_local_ncc_forward) * 100:.3f} +- {np.std(avg_local_ncc_forward) * 100:.3f} (Original Local NCC: {avg_init_local_ncc * 100:.3f})')
    logger.debug(f'Average Forward Percentage of Negative Jacobian Determinants: {np.mean(avg_neg_jac_dets_forward)}+-{np.std(avg_neg_jac_dets_forward)}')
    logger.debug(f'Reverse Absolute Difference: {avg_difference_mean_reverse:.6f} +- {avg_difference_std_reverse:.6f}')
    logger.debug(f'Average Reverse NCC: {np.mean(avg_ncc_reverse) * 100:.3f} +- {np.std(avg_ncc_reverse) * 100:.3f} (Original NCC: {avg_init_ncc * 100:.3f})')
    logger.debug(f'Average Reverse Local NCC: {np.mean(avg_local_ncc_forward) * 100:.3f} +- {np.std(avg_local_ncc_forward) * 100:.3f} (Original Local NCC: {avg_init_local_ncc * 100:.3f})')
    logger.debug(f'Average Reverse Percentage of Negative Jacobian Determinants: {np.mean(avg_neg_jac_dets_reverse)}+-{np.std(avg_neg_jac_dets_reverse)}')
    logger.debug(f'Average Forward DICE score: {np.mean(avg_forward_dice) * 100:.3f}+-{np.std(avg_forward_dice) * 100:.3f} ({np.mean(avg_forward_dice2) * 100:.3f} +- {np.std(avg_forward_dice2) * 100:.3f})')
    logger.debug(f'Average Reverse DICE score: {np.mean(avg_reverse_dice) * 100:.3f} +- {np.std(avg_reverse_dice) * 100:.3f} ({np.mean(avg_reverse_dice2) * 100:.3f} +- {np.std(avg_reverse_dice2) * 100:.3f})')
    logger.debug(f'Average Forward SSIM: {np.mean(avg_forward_ssim) * 100:.3f} +- {np.std(avg_forward_ssim) * 100:.3f}')
    logger.debug(f'Average Reverse SSIM: {np.mean(avg_reverse_ssim) * 100:.3f} +- {np.std(avg_reverse_ssim) * 100:.3f}')
    
    evaluate_semi_group_property(network, I_, J_, xyz_, getattr(config, 'save_evals_path'))
