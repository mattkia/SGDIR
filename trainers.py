import os
import time
import torch
import logging

import numpy as np
import torch.optim as optim

from tqdm import tqdm
from typing import Dict
from tabulate import tabulate
from torch.utils.data import DataLoader

from metrics import TRE
from metrics import HD95
from metrics import Dice
from metrics import ASSD
from metrics import SSIM3d
from metrics import SDLogJ
from metrics import NCCLoss
from metrics import SurfaceDice
from metrics import DicePerStructure
from metrics import JacobianDeterminant

from data import get_dataset

from utils import warp
from utils import rolling_dice

from visualization import save_grid
from visualization import save_mask
from visualization import save_image
from visualization import flow_snapshots
from visualization import animate_warping
from visualization import save_vector_field
from visualization import save_grid_overlayed_image

from models.builder import build_model


class DiceTrainer:
    """
    A trainer where the data segmentation maps are available and
    the evaluation is done based on the dice score and HD95 metric.
    """
    def __init__(self, config: Dict):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        data_config = config.get('data')
        model_config = config.get('model')
        training_config = config.get('training')
        loss_config = config.get('loss')
        
        self.batch_size = training_config.get('batch_size', 1)
        self.res_levels = training_config.get('res_levels', 8)

        self.lr = float(training_config.get('lr', 1e-4))
        self.epochs = training_config.get('epochs', 500)
        self.eval_interval = training_config.get('eval_interval', 1)
        self.ckpt_path = training_config.get('checkpoints_path')

        # creating required directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # defining the logger
        logdir = training_config.get('logdir')
        logging.basicConfig(filename=logdir,
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        logging.getLogger('PIL').setLevel(logging.ERROR)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # loading the dataset and dataloader
        self.dataset = get_dataset(config=data_config,
                                   train=True)
        self.loader = DataLoader(self.dataset,
                                 shuffle=True,
                                 batch_size=self.batch_size)

        # loading the validation dataset
        self.val_dataset = get_dataset(config=data_config,
                                       train=False,
                                       val=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     shuffle=False,
                                     batch_size=self.batch_size)

        # initializing the network
        self.network = build_model(model_config).to(self.device)

        # setting up the optimizer
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.lr)

        # setting loss factors
        self.image_loss_coeff = float(loss_config.get('image_loss_coeff', 1.))
        self.ss_loss_coeff = float(loss_config.get('semigroup_coeff'))

        # defining the metrics
        self.hd95 = HD95()
        self.assd  = ASSD()
        self.ssim = SSIM3d()
        self.dice1 = Dice(structured=True)
        self.dice2 = Dice(structured=False)
        # self.dice2 = SurfaceDice()
        self.jac_det = JacobianDeterminant()

    def run(self):
        print('[*] Training on Continuous Semigroup Regularization...')
        self.logger.debug('Training with semigroup regularization started...')

        best_dice = 0.
        iters = tqdm(range(self.epochs))
        iters.set_description('Loss: Inf')

        for epoch in iters:
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
            avg_assd = 0.
            
            for sample in self.loader:
                if self.res_levels > 1:
                    res = np.sort((0.95 - 0.05) * np.random.rand(self.res_levels - 1) + 0.05)
                    res = np.concatenate((res, [1.0]))
                else:
                    res = [1.0]
                s = time.time()
                for level in range(self.res_levels):
                    fixed = sample[0].to(self.device)
                    moving = sample[1].to(self.device)
                    grid = sample[2].to(self.device)
                    fixed_seg = sample[3].to(self.device)
                    moving_seg = sample[4].to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    loss, ss_loss = self.network.loss_flow(fixed, moving, grid, res[level])
                    
                    total_loss = self.image_loss_coeff * loss + self.ss_loss_coeff * ss_loss

                    total_loss.backward()
                    self.optimizer.step()
                    
                    avg_img_loss += loss.item()
                    avg_ss_loss += ss_loss.item()
                    avg_loss += total_loss.item()
                with torch.no_grad():
                    deformation = self.network(fixed,
                                               moving,
                                               grid,
                                               torch.ones(1, device=self.device))

                    moving_seg_warped = warp(moving_seg, deformation, True)

                    tf_dice = self.dice1(fixed_seg, moving_seg_warped)

                    avg_tf_dice += tf_dice * 100
                    avg_tf_det += self.jac_det(deformation)
            
            avg_img_loss /= (self.res_levels * len(self.loader))
            avg_ss_loss /= (self.res_levels * len(self.loader))
            avg_loss /= (self.res_levels * len(self.loader))
            avg_tf_dice /= len(self.loader)
            avg_tf_det /= len(self.loader)

            for sample in self.val_loader:
                fixed = sample[0].to(self.device)
                moving = sample[1].to(self.device)
                grid = sample[2].to(self.device)
                fixed_seg = sample[3].to(self.device)
                moving_seg = sample[4].to(self.device)

                with torch.no_grad():
                    deformation = self.network(fixed,
                                               moving,
                                               grid,
                                               torch.ones(1, device=self.device))

                    moving_warped = warp(moving, deformation)
                    moving_seg_warped = warp(moving_seg, deformation, True)

                    f_dice = self.dice1(fixed_seg, moving_seg_warped)
                    f_dice2 = self.dice2(fixed_seg.clone(), moving_seg_warped.clone())

                    avg_f_dice += f_dice * 100
                    avg_f_dice2 += f_dice2 * 100
                    avg_f_det += self.jac_det(deformation)
                    avg_hd += self.hd95(fixed_seg, moving_seg_warped)
                    avg_assd += self.assd(fixed_seg, moving_seg_warped)
                    avg_f_ssim += self.ssim(fixed, moving_warped) * 100
                
            avg_f_dice /= len(self.val_loader)
            avg_f_dice2 /= len(self.val_loader)
            avg_f_det /= len(self.val_loader)
            avg_f_ssim /= len(self.val_loader)
            avg_hd /= len(self.val_loader)
            
            e_time = time.time()
            elapsed = (e_time - s_time) / 60.
            
            if epoch % self.eval_interval == 0 or epoch == self.epochs - 1:
                iters.set_description(f'Loss: {avg_loss:.6f}')
                log = f'Epoch:{epoch}, Image Loss: {avg_img_loss:.4f}, Contraint Loss: {avg_ss_loss * self.ss_loss_coeff:.4f}, '
                log += f'Train Dice: {avg_tf_dice:.2f}, Train Det: {avg_tf_det:.4f}, '
                log += f'Val Dice: {avg_f_dice:.2f} ({avg_f_dice2:.2f}), Val Det: {avg_f_det:.4f}, '
                log += f'Val SSIM: {avg_f_ssim:.2f}, Val HD: {avg_hd:.2f}, Elapsed: {elapsed:.2f}m'
                
                if avg_f_dice2 > best_dice:
                    best_dice = avg_f_dice2
                    torch.save(self.network.state_dict(), self.ckpt_path)
                    log += ' [CHCKPNT]'
                    
                self.logger.debug(log)


class TRETrainer:
    """
    A trainer where the data landmarks/keypoints are available and
    the evaluation is done based on the TRE score.
    """
    def __init__(self, config: Dict):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        data_config = config.get('data')
        model_config = config.get('model')
        training_config = config.get('training')
        loss_config = config.get('loss')

        self.batch_size = training_config.get('batch_size', 1)
        self.res_levels = training_config.get('res_levels', 8)

        self.lr = float(training_config.get('lr', 1e-4))
        self.epochs = training_config.get('epochs', 500)
        self.eval_interval = training_config.get('eval_interval', 1)
        self.ckpt_path = training_config.get('checkpoints_path')

        # creating required directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # defining the logger
        logdir = training_config.get('logdir')
        logging.basicConfig(filename=logdir,
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        logging.getLogger('PIL').setLevel(logging.ERROR)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # loading the dataset and dataloader
        self.dataset = get_dataset(config=data_config,
                                   train=True)
        self.loader = DataLoader(self.dataset,
                                 shuffle=True,
                                 batch_size=self.batch_size)

        # loading the validation dataset
        self.val_dataset = get_dataset(config=data_config,
                                       train=False,
                                       val=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     shuffle=False,
                                     batch_size=self.batch_size)

        # initializing the network
        self.network = build_model(model_config).to(self.device)

        # setting up the optimizer
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.lr)

        # setting loss factors
        self.image_loss_coeff = float(loss_config.get('image_loss_coeff', 1.))
        self.ss_loss_coeff = float(loss_config.get('semigroup_coeff'))

        # defining the metrics
        self.tre = TRE()
        self.ssim = SSIM3d()
        self.jac_det = JacobianDeterminant()

    def run(self):
        print('[*] Training on Continuous Semigroup Regularization...')
        self.logger.debug('Training with semigroup regularization started...')

        best_tre = float('inf')
        iters = tqdm(range(self.epochs))
        iters.set_description('Loss: Inf')

        for epoch in iters:
            s_time = time.time()
            avg_loss = 0.
            avg_img_loss = 0.
            avg_ss_loss = 0.
            avg_f_tre = 0.
            avg_f_det = 0.
            avg_tf_tre = 0.
            avg_tf_det = 0.
            avg_f_ssim = 0.
            
            for sample in self.loader:
                if self.res_levels > 1:
                    res = np.sort((0.95 - 0.05) * np.random.rand(self.res_levels - 1) + 0.05)
                    res = np.concatenate((res, [1.0]))
                else:
                    res = [1.0]
                
                for level in range(self.res_levels):
                    fixed = sample[0].to(self.device)
                    moving = sample[1].to(self.device)
                    grid = sample[2].to(self.device)
                    fixed_kp = sample[3].to(self.device)
                    moving_kp = sample[4].to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    loss, ss_loss = self.network.loss_flow(fixed, moving, grid, res[level])
                    
                    total_loss = self.image_loss_coeff * loss + self.ss_loss_coeff * ss_loss
                    
                    total_loss.backward()
                    self.optimizer.step()
                    
                    avg_img_loss += loss.item()
                    avg_ss_loss += ss_loss.item()
                    avg_loss += total_loss.item()
                
                with torch.no_grad():
                    deformation = self.network(fixed,
                                               moving,
                                               grid,
                                               torch.ones(1, device=self.device))

                    tf_tre = self.tre(fixed_kp.clone(),
                                      moving_kp.clone(),
                                      deformation.clone(),
                                      grid.clone())[0]

                    avg_tf_tre += tf_tre
                    avg_tf_det += self.jac_det(deformation)
            
            avg_img_loss /= (self.res_levels * len(self.loader))
            avg_ss_loss /= (self.res_levels * len(self.loader))
            avg_loss /= (self.res_levels * len(self.loader))
            avg_tf_tre /= len(self.loader)
            avg_tf_det /= len(self.loader)

            for sample in self.val_loader:
                fixed = sample[0].to(self.device)
                moving = sample[1].to(self.device)
                grid = sample[2].to(self.device)
                fixed_kp = sample[3].to(self.device)
                moving_kp = sample[4].to(self.device)

                with torch.no_grad():
                    deformation = self.network(fixed,
                                               moving,
                                               grid,
                                               torch.ones(1, device=self.device))

                    f_tre = self.tre(fixed_kp.clone(),
                                     moving_kp.clone(),
                                     deformation.clone(),
                                     grid.clone())[0]

                    moving_warped = warp(moving, deformation)

                    avg_f_tre += f_tre
                    avg_f_det += self.jac_det(deformation)
                    avg_f_ssim += self.ssim(fixed, moving_warped) * 100
                
            avg_f_tre /= len(self.val_loader)
            avg_f_det /= len(self.val_loader)
            avg_f_ssim /= len(self.val_loader)
            
            e_time = time.time()
            elapsed = (e_time - s_time) / 60.
            
            if epoch % self.eval_interval == 0 or epoch == self.epochs - 1:
                iters.set_description(f'Loss: {avg_loss:.6f}')
                log = f'Epoch:{epoch}, Image Loss: {avg_img_loss:.4f}, Contraint Loss: {avg_ss_loss * self.ss_loss_coeff:.4f}, '
                log += f'Train TRE: {avg_tf_tre:.4f}, Train Det: {avg_tf_det:.4f}, '
                log += f'Val TRE: {avg_f_tre:.4f}, Val Det: {avg_f_det:.4f}, '
                log += f'Val SSIM: {avg_f_ssim:.2f}, Elapsed: {elapsed:.2f}m'
                
                if avg_f_tre < best_tre:
                    best_tre = avg_f_tre
                    torch.save(self.network.state_dict(), self.ckpt_path)
                    log += ' [CHCKPNT]'
                    
                self.logger.debug(log)


class DiceTester:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        data_config = config.get('data')
        model_config = config.get('model')
        training_config = config.get('training')
        test_config = config.get('test')

        self.ckpt_path = training_config.get('checkpoints_path')
        self.weights = torch.load(self.ckpt_path,
                                  weights_only=True)
        
        self.save_evals = test_config.get('save_evals', False)
        self.save_evals_path = test_config.get('save_evals_path', '.')

        self.batch_size = 1
        self.down_factor = test_config.get('down_factor', 1)

        # creating required directories
        os.makedirs(self.save_evals_path, exist_ok=True)

        # defining the logger
        logdir = test_config.get('logdir')
        logging.basicConfig(filename=logdir,
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        logging.getLogger('PIL').setLevel(logging.ERROR)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # loading the dataset and dataloader
        self.dataset = get_dataset(config=data_config, train=False)
        self.loader = DataLoader(self.dataset,
                                 shuffle=False,
                                 batch_size=self.batch_size)

        # initializing the network
        self.network = build_model(model_config).to(self.device)
        self.network.load_state_dict(self.weights)

        # defining the metrics
        self.hd95 = HD95()
        self.assd = ASSD()
        self.ssim = SSIM3d()
        self.dice1 = Dice(structured=True)
        self.dice2 = Dice(structured=False)
        self.sdlogj = SDLogJ()
        self.struct_dice = DicePerStructure()
        self.jac_det = JacobianDeterminant()

    @torch.no_grad()
    def run(self):
        ncc = NCCLoss(win=11)

        print('[*] Evaluation started...')
        self.logger.debug(f'Evaluation started on {len(self.dataset)} samples...')
        avg_forward_jac_det = []
        avg_backward_jac_det = []
        avg_forward_sdlogj = []
        avg_backward_sdlogj = []
        avg_forward_ncc = []
        avg_backward_ncc = []
        avg_forward_diff = []
        avg_backward_diff = []
        avg_forward_dice = []
        avg_backward_dice = []
        avg_forward_dice2 = []
        avg_backward_dice2 = []
        avg_forward_ssim = []
        avg_backward_ssim = []
        avg_forward_hd95 = []
        avg_backward_hd95 = []
        avg_forward_assd = []
        avg_backward_assd = []
        avg_forward_struct_dice = []
        avg_backward_struct_dice = []
        
        for batch_id, sample in enumerate(tqdm(self.loader)):            
            fixed = sample[0].to(self.device)
            moving = sample[1].to(self.device)
            grid = sample[2].to(self.device)
            fixed_seg = sample[3].to(self.device)
            moving_seg = sample[4].to(self.device)
            
            t = torch.ones(1, dtype=torch.float32).to(self.device)
            forward_deformation = self.network(fixed, moving, grid, t)
            backward_deformation = self.network(fixed, moving, grid, -t)

            moving_warped = warp(moving, forward_deformation)
            moving_seg_warped = warp(moving_seg, forward_deformation, True)

            fixed_warped = warp(fixed, backward_deformation)
            fixed_seg_warped = warp(fixed_seg, backward_deformation, True)
            
            forward_jac_det = self.jac_det(forward_deformation).item()
            backward_jac_det = self.jac_det(backward_deformation).item()

            forward_sdlogj = self.sdlogj(forward_deformation).item()
            backward_sdlogj = self.sdlogj(backward_deformation).item()

            forward_ncc = ncc(fixed, moving_warped).item()
            backward_ncc = ncc(moving, fixed_warped).item()

            forward_difference = (moving_warped - fixed).abs()
            backward_difference = (fixed_warped - moving).abs()

            forward_dice = self.dice1(fixed_seg, moving_seg_warped)
            backward_dice = self.dice1(moving_seg, fixed_seg_warped)
            forward_dice2 = self.dice2(fixed_seg.clone(), moving_seg_warped.clone())
            backward_dice2 = self.dice2(moving_seg.clone(), fixed_seg_warped.clone())

            forward_struct_dice = self.struct_dice(fixed_seg, moving_seg_warped)
            backward_struct_dice = self.struct_dice(moving_seg, fixed_seg_warped)

            forward_hd95 = self.hd95(fixed_seg, moving_seg_warped)
            backward_hd95 = self.hd95(moving_seg, fixed_seg_warped)
            
            forward_assd = self.assd(fixed_seg, moving_seg_warped)
            backward_assd = self.assd(moving_seg, fixed_seg_warped)

            avg_forward_dice.append(forward_dice.item())
            avg_backward_dice.append(backward_dice.item())
            avg_forward_dice2.append(forward_dice2.item())
            avg_backward_dice2.append(backward_dice2.item())

            avg_forward_struct_dice.append(forward_struct_dice)
            avg_backward_struct_dice.append(backward_struct_dice)

            avg_forward_ssim.append(self.ssim(fixed, moving_warped).item())
            avg_backward_ssim.append(self.ssim(moving, fixed_warped).item())

            avg_forward_diff.append(forward_difference.mean().item())
            avg_backward_diff.append(backward_difference.mean().item())
            
            avg_forward_jac_det.append(forward_jac_det)
            avg_backward_jac_det.append(backward_jac_det)

            avg_forward_sdlogj.append(forward_sdlogj)
            avg_backward_sdlogj.append(backward_sdlogj)

            avg_forward_ncc.append(forward_ncc)
            avg_backward_ncc.append(backward_ncc)

            avg_forward_hd95.append(forward_hd95)
            avg_backward_hd95.append(backward_hd95)
            
            avg_forward_assd.append(forward_assd)
            avg_backward_assd.append(backward_assd)

            if self.save_evals:
                batch_path = os.path.join(self.save_evals_path, f'batch_{batch_id}')
                os.makedirs(batch_path, exist_ok=True)

                rolling_dice(self.network,
                             fixed,
                             moving,
                             fixed_seg,
                             moving_seg,
                             grid,
                             batch_path,
                             num_frames=50)

                save_image(fixed,
                           os.path.join(batch_path, 'fixed.png'),
                           cmap='bone')
                save_image(moving,
                           os.path.join(batch_path, 'moving.png'),
                           cmap='bone')
                save_image(fixed_warped,
                           os.path.join(batch_path, 'fixed_warped.png'),
                           cmap='bone')
                save_image(moving_warped,
                           os.path.join(batch_path, 'moving_warped.png'),
                           cmap='bone')
                save_image((fixed - moving).abs(),
                           os.path.join(batch_path, 'diff_before.png'),
                           cmap='bone')
                save_image((fixed - moving_warped).abs(),
                           os.path.join(batch_path, 'diff_after.png'),
                           cmap='bone')

                save_grid(forward_deformation,
                          os.path.join(batch_path, 'forward_grid.png'),
                          down_factor=self.down_factor)
                save_grid(backward_deformation,
                          os.path.join(batch_path, 'backward_grid.png'),
                          down_factor=self.down_factor)
                
                save_grid(forward_deformation,
                          os.path.join(batch_path, 'forward_grid_full.png'),
                          down_factor=1)
                save_grid(backward_deformation,
                          os.path.join(batch_path, 'backward_grid_full.png'),
                          down_factor=1)

                save_mask(moving_seg_warped,
                          os.path.join(batch_path, 'moving_seg_warped.png'),
                          num_classes=35)
                save_mask(fixed_seg_warped,
                          os.path.join(batch_path, 'fixed_seg_warped.png'),
                          num_classes=35)

                save_grid_overlayed_image(moving_warped,
                                          backward_deformation,
                                          os.path.join(batch_path, 'moving_grid_warped.png'),
                                          grid_downfactor=self.down_factor,
                                          cmap='gray')

                flow_snapshots(self.network,
                               fixed, moving, grid,
                               save_path=batch_path,
                               num_frames=7)

                animate_warping(self.network,
                                fixed, moving, grid,
                                down_factor=self.down_factor,
                                save_path=batch_path,
                                num_frames=50,
                                overlay_grid=True,
                                cmap='bone')

                animate_warping(self.network,
                                fixed, moving, grid,
                                down_factor=self.down_factor,
                                save_path=batch_path,
                                num_frames=50,
                                overlay_grid=False,
                                cmap='bone')

        mean_f_diff = np.mean(avg_forward_diff)
        std_f_diff = np.std(avg_forward_diff)
        mean_f_ncc = np.mean(avg_forward_ncc)
        std_f_ncc = np.std(avg_forward_ncc)
        mean_f_jac_det = np.mean(avg_forward_jac_det)
        std_f_jac_det = np.std(avg_forward_jac_det)
        mean_f_sdlogj = np.mean(avg_forward_sdlogj)
        std_f_sdlogj = np.std(avg_forward_sdlogj)
        mean_f_dice = np.mean(avg_forward_dice)
        mean_f_dice2 = np.mean(avg_forward_dice2)
        std_f_dice = np.std(avg_forward_dice)
        std_f_dice2 = np.std(avg_forward_dice2)
        mean_f_ssim = np.mean(avg_forward_ssim)
        std_f_ssim = np.std(avg_forward_ssim)
        mean_f_hd95 = np.mean(avg_forward_hd95)
        std_f_hd95 = np.std(avg_forward_hd95)
        mean_f_assd = np.mean(avg_forward_assd)
        std_f_assd = np.std(avg_forward_assd)

        mean_b_diff = np.mean(avg_backward_diff)
        std_b_diff = np.std(avg_backward_diff)
        mean_b_ncc = np.mean(avg_backward_ncc)
        std_b_ncc = np.std(avg_backward_ncc)
        mean_b_jac_det = np.mean(avg_backward_jac_det)
        std_b_jac_det = np.std(avg_backward_jac_det)
        mean_b_sdlogj = np.mean(avg_backward_sdlogj)
        std_b_sdlogj = np.std(avg_backward_sdlogj)
        mean_b_dice = np.mean(avg_backward_dice)
        mean_b_dice2 = np.mean(avg_backward_dice2)
        std_b_dice = np.std(avg_backward_dice)
        std_b_dice2 = np.std(avg_backward_dice2)
        mean_b_ssim = np.mean(avg_backward_ssim)
        std_b_ssim = np.std(avg_backward_ssim)
        mean_b_hd95 = np.mean(avg_backward_hd95)
        std_b_hd95 = np.std(avg_backward_hd95)
        mean_b_assd = np.mean(avg_backward_assd)
        std_b_assd = np.std(avg_backward_assd)

        performance_summary = [
            ['Metric', 'Forward', 'Backward'],
            ['Abs. Diff.', f'{mean_f_diff:.4f} +- {std_f_diff:.4f}', f'{mean_b_diff:.4f} +- {std_b_diff:.4f}'],
            ['Avg. NCC', f'{mean_f_ncc:.4f} +- {std_f_ncc:.4f}', f'{mean_b_ncc:.4f} +- {std_b_ncc:.4f}'],
            ['Folding %', f'{mean_f_jac_det:.4f} +- {std_f_jac_det:.4f}', f'{mean_b_jac_det:.4f} +- {std_b_jac_det:.4f}'],
            ['SDLogJ', f'{mean_f_sdlogj:.4f} +- {std_f_sdlogj:.4f}', f'{mean_b_sdlogj:.4f} +- {std_b_sdlogj:.4f}'],
            ['Structured Dice', f'{mean_f_dice:.4f} +- {std_f_dice:.4f}', f'{mean_b_dice:.4f} +- {std_b_dice:.4f}'],
            ['Dice', f'{mean_f_dice2:.4f} +- {std_f_dice2:.4f}', f'{mean_b_dice2:.4f} +- {std_b_dice2:.4f}'],
            ['SSIM', f'{mean_f_ssim:.4f} +- {std_f_ssim:.4f}', f'{mean_b_ssim:.4f} +- {std_b_ssim:.4f}'],
            ['HD95', f'{mean_f_hd95:.4f} +- {std_f_hd95:.4f}', f'{mean_b_hd95:.4f} +- {std_b_hd95:.4f}'],
            ['ASSD', f'{mean_f_assd:.4f} +- {std_f_assd:.4f}', f'{mean_b_assd:.4f} +- {std_b_assd:.4f}'],
        ]
        
        table = tabulate(performance_summary, headers='firstrow', tablefmt='grid')

        # compute per-structure-dice averages
        f_struct_dice = torch.stack(avg_forward_struct_dice, axis=1).cpu()
        b_struct_dice = torch.stack(avg_backward_struct_dice, axis=1).cpu()

        mean_f_struct_dice = f_struct_dice.mean(axis=1)
        std_f_struct_dice = f_struct_dice.std(axis=1)
        mean_b_struct_dice = b_struct_dice.mean(axis=1)
        std_b_struct_dice = b_struct_dice.std(axis=1)

        structured_dice_summary = [
            ['Metric/Label', *range(len(mean_f_struct_dice))],
            ['Forward Mean', *mean_f_struct_dice.tolist()],
            ['Forward Std', *std_f_struct_dice.tolist()],
            ['Backward Mean', *mean_b_struct_dice.tolist()],
            ['Backward Std', *std_b_struct_dice.tolist()],
        ]

        dice_table = tabulate(structured_dice_summary, headers='firstrow', tablefmt='grid')

        self.logger.debug('Evaluation Finished')
        self.logger.debug('\n' + table + '\n' + dice_table)

        # saving the structured dices for plot purposes
        np.savetxt(os.path.join(self.save_evals_path, 'f_struc_dice.txt'), f_struct_dice.numpy())
        np.savetxt(os.path.join(self.save_evals_path, 'b_struc_dice.txt'), b_struct_dice.numpy())


class TRETester:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        data_config = config.get('data')
        model_config = config.get('model')
        training_config = config.get('training')
        test_config = config.get('test')

        self.ckpt_path = training_config.get('checkpoints_path')
        self.weights = torch.load(self.ckpt_path,
                                  weights_only=True)
        
        self.save_evals = test_config.get('save_evals', False)
        self.save_evals_path = test_config.get('save_evals_path', '.')

        self.batch_size = 1
        self.down_factor = test_config.get('down_factor', 1)

        # creating required directories
        os.makedirs(self.save_evals_path, exist_ok=True)

        # defining the logger
        logdir = test_config.get('logdir')
        logging.basicConfig(filename=logdir,
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        logging.getLogger('PIL').setLevel(logging.ERROR)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # loading the dataset and dataloader
        self.dataset = get_dataset(config=data_config, train=False)
        self.loader = DataLoader(self.dataset,
                                 shuffle=False,
                                 batch_size=self.batch_size)

        # initializing the network
        self.network = build_model(model_config).to(self.device)
        self.network.load_state_dict(self.weights)

        # defining the metrics
        self.tre = TRE()
        self.ssim = SSIM3d()
        self.jac_det = JacobianDeterminant()

    @torch.no_grad()
    def run(self):
        ncc = NCCLoss(win=11)

        print('[*] Evaluation started...')
        self.logger.debug(f'Evaluation started on {len(self.dataset)} samples...')
        avg_forward_jac_det = []
        avg_backward_jac_det = []
        avg_forward_ncc = []
        avg_backward_ncc = []
        avg_forward_diff = []
        avg_backward_diff = []
        avg_forward_tre = []
        avg_backward_tre = []
        avg_forward_ssim = []
        avg_backward_ssim = []
        
        for batch_id, sample in enumerate(tqdm(self.loader)): 
            fixed = sample[0].to(self.device)
            moving = sample[1].to(self.device)
            grid = sample[2].to(self.device)
            fixed_kp = sample[3].to(self.device)
            moving_kp = sample[4].to(self.device)
            
            t = torch.ones(1, dtype=torch.float32).to(self.device)
            forward_deformation = self.network(fixed, moving, grid, t)
            backward_deformation = self.network(fixed, moving, grid, -t)
            
            moving_warped = warp(moving, forward_deformation)
            fixed_warped = warp(fixed, backward_deformation)
            
            forward_jac_det = self.jac_det(forward_deformation)
            backward_jac_det = self.jac_det(backward_deformation)
            
            forward_ncc = ncc(fixed, moving_warped).item()
            backward_ncc = ncc(moving, fixed_warped).item()
            
            forward_difference = (moving_warped - fixed).abs()
            backward_difference = (fixed_warped - moving).abs()
            
            forward_tre = self.tre(fixed_kp.clone(),
                                   moving_kp.clone(),
                                   forward_deformation.clone(),
                                   grid.clone())[0]
            backward_tre = self.tre(moving_kp.clone(),
                                    fixed_kp.clone(),
                                    backward_deformation.clone(),
                                    grid.clone())[0]
            
            avg_forward_tre.append(forward_tre.item())
            avg_backward_tre.append(backward_tre.item())
            
            avg_forward_ssim.append(self.ssim(fixed, moving_warped).item())
            avg_backward_ssim.append(self.ssim(moving, fixed_warped).item())
            
            avg_forward_diff.append(forward_difference.mean().item())
            avg_backward_diff.append(backward_difference.mean().item())
            
            avg_forward_jac_det.append(forward_jac_det.item())
            avg_backward_jac_det.append(backward_jac_det.item())

            avg_forward_ncc.append(forward_ncc)
            avg_backward_ncc.append(backward_ncc)
            
            if self.save_evals:
                batch_path = os.path.join(self.save_evals_path, f'batch_{batch_id}')
                os.makedirs(batch_path, exist_ok=True)

                save_image(fixed,
                           os.path.join(batch_path, 'fixed.png'),
                           cmap='bone')
                save_image(moving,
                           os.path.join(batch_path, 'moving.png'),
                           cmap='bone')
                save_image(fixed_warped,
                           os.path.join(batch_path, 'fixed_warped.png'),
                           cmap='bone')
                save_image(moving_warped,
                           os.path.join(batch_path, 'moving_warped.png'),
                           cmap='bone')
                save_image((fixed - moving).abs(),
                           os.path.join(batch_path, 'diff_before.png'),
                           cmap='bone')
                save_image((fixed - moving_warped).abs(),
                           os.path.join(batch_path, 'forward_diff.png'),
                           cmap='bone')
                save_image((fixed_warped - moving).abs(),
                           os.path.join(batch_path, 'backward_diff.png'),
                           cmap='bone')
                save_vector_field(forward_deformation,
                                  os.path.join(batch_path, 'field.png'),
                                  moving_kp,
                                  grid)

        mean_f_diff = np.mean(avg_forward_diff)
        std_f_diff = np.std(avg_forward_diff)
        mean_f_ncc = np.mean(avg_forward_ncc)
        std_f_ncc = np.std(avg_forward_ncc)
        mean_f_jac_det = np.mean(avg_forward_jac_det)
        std_f_jac_det = np.std(avg_forward_jac_det)
        mean_f_tre = np.mean(avg_forward_tre)
        std_f_tre = np.std(avg_forward_tre)
        mean_f_ssim = np.mean(avg_forward_ssim)
        std_f_ssim = np.std(avg_forward_ssim)

        mean_b_diff = np.mean(avg_backward_diff)
        std_b_diff = np.std(avg_backward_diff)
        mean_b_ncc = np.mean(avg_backward_ncc)
        std_b_ncc = np.std(avg_backward_ncc)
        mean_b_jac_det = np.mean(avg_backward_jac_det)
        std_b_jac_det = np.std(avg_backward_jac_det)
        mean_b_tre = np.mean(avg_backward_tre)
        std_b_tre = np.std(avg_backward_tre)
        mean_b_ssim = np.mean(avg_backward_ssim)
        std_b_ssim = np.std(avg_backward_ssim)

        performance_summary = [
            ['Metric', 'Forward', 'Backward'],
            ['Abs. Diff.', f'{mean_f_diff:.4f} +- {std_f_diff:.4f}', f'{mean_b_diff:.4f} +- {std_b_diff:.4f}'],
            ['Avg. NCC', f'{mean_f_ncc:.4f} +- {std_f_ncc:.4f}', f'{mean_b_ncc:.4f} +- {std_b_ncc:.4f}'],
            ['Folding %', f'{mean_f_jac_det:.4f} +- {std_f_jac_det:.4f}', f'{mean_b_jac_det:.4f} +- {std_b_jac_det:.4f}'],
            ['TRE', f'{mean_f_tre:.4f} +- {std_f_tre:.4f}', f'{mean_b_tre:.4f} +- {std_b_tre:.4f}'],
            ['SSIM', f'{mean_f_ssim:.4f} +- {std_f_ssim:.4f}', f'{mean_b_ssim:.4f} +- {std_b_ssim:.4f}'],
        ]
        
        table = tabulate(performance_summary, headers='firstrow', tablefmt='grid')
        self.logger.debug('Evaluation Finished')
        self.logger.debug('\n'+table)
        # evaluate_semi_group_property(network, fixed, moving, grid, getattr(config, 'save_evals_path'))
