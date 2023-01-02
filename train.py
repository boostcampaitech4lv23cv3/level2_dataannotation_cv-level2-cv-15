import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import numpy as np
import random
import wandb

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed):
    
    seed_everything(seed)
    
    wandb.init(project="Data_make",
                entity='fullhouse',
                name="HJ_CosineAnnealingLR"
    )

    train_dataset = SceneTextDataset(data_dir, split='K-fold_train1', image_size=image_size, crop_size=input_size, train_transform=True)
    val_dataset = SceneTextDataset(data_dir, split='K-fold_val1', image_size=image_size, crop_size=input_size, train_transform=False)
    train_dataset = EASTDataset(train_dataset)
    val_dataset = EASTDataset(val_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=1)
    wandb.watch(model, log='all')
    
    # early_stopping : 17번의 epoch 연속으로 val loss 미개선 시에 조기 종료
    patience = 17

    best_val_loss = np.inf
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        model.train()
        with tqdm(total=train_num_batches) as pbar:
            for step, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description('[Epoch {} train]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                train_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }

                if step % 20 == 0:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/Cls loss": extra_info['cls_loss'],
                        "train/Angle loss": extra_info['angle_loss'],
                        "train/IoU loss": extra_info['iou_loss'],
                    })

                pbar.set_postfix(train_dict)
        
        wandb.log({
            "Charts/learning_rate": optimizer.param_groups[0]['lr']})
        
        print('Mean train loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        model.eval()
        with torch.no_grad():
            with tqdm(total=val_num_batches) as pbar:
                for step, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(val_loader):
                    pbar.set_description('[Epoch {} val]'.format(epoch + 1))

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    
                    loss_val = loss.item()
                    epoch_loss += loss_val

                    pbar.update(1)
                    
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss'], 
                    }
                    
                    if step % 20 == 0:
                        wandb.log({
                            "val/loss": loss_val,
                            "val/Cls loss": extra_info['cls_loss'],
                            "val/Angle loss": extra_info['angle_loss'],
                            "val/IoU loss": extra_info['iou_loss'],
                        })

                    pbar.set_postfix(val_dict)
            
            val_loss = epoch_loss / val_num_batches
            scheduler.step(val_loss)
            
            print('Mean val loss: {:.4f} | Elapsed time: {}'.format(
                val_loss, timedelta(seconds=time.time() - epoch_start)))

            if val_loss < best_val_loss:
                print('trigger times: 0')
                trigger_times = 0
                best_val_loss = val_loss
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)
                print('Saving best.pth ...')
                ckpt_fpath = osp.join(model_dir, 'best.pth')
                torch.save(model.state_dict(), ckpt_fpath)
            else:
                trigger_times += 1
                print('Trigger Times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return model

            if (epoch + 1) % save_interval == 0:
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                ckpt_fpath = osp.join(model_dir, 'latest.pth')
                torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
