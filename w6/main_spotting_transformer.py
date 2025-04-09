#!/usr/bin/env python3
"""
File containing the main training script.
"""

# Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

# Local imports
from util.io import load_json, store_json
from util.eval_spotting_trans import evaluate
from dataset.datasets import get_datasets
from model.model_spotting_transformer import TransformerModel  

from utils import get_next_experiment_folder
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def get_args():
    # Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    # Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']
    
    # Add transformer-specific parameters
    if 'transformer' in config:
        args.transformer = config['transformer']
        args.temporal_model = config.get('temporal_model', 'transformer')
        args.optimizer = config.get('optimizer', 'adamw')
        args.weight_decay = config.get('weight_decay', 0.01)
        args.use_amp = config.get('use_amp', True)

    if not args.only_test:
        save_dir = config['save_dir'] + '/' + args.model
        args.save_dir = get_next_experiment_folder(save_dir, name="exp")
    else:
        args.save_dir = config['save_dir'] + '/' + args.model
    
    args.early_stop_patience = config['early_stop_patience']
    args.early_stop_delta = config['early_stop_delta']
    args.store_to_wandb = config['store_to_wandb']

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])

def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = '/ghome/c5mcv04/w6/spotting/config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    if not args.only_test:
        ckpt_dir = os.path.join(args.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        ckpt_dir = os.path.join(args.chkpoint_dir, "checkpoints")
    print(ckpt_dir)
    
    if args.store_to_wandb:
        wandb.init(project="ball-action-detection", entity="c6mcv06", name=f"{os.path.basename(ckpt_dir)}")

    # Get datasets
    classes, train_data, val_data, val_data_video, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    model = TransformerModel(args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Training loop
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = float('inf')
        epoch = 0
        epochs_without_improvement = 0
        print('START TRAINING EPOCHS')
        
        for epoch in range(epoch, num_epochs):
            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)

            better = False
            if val_loss < best_criterion - args.early_stop_delta:
                best_criterion = val_loss
                better = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            print(f'[Epoch {epoch}] Train loss: {train_loss:0.5f} Val loss: {val_loss:0.5f}')
            if args.store_to_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch + 1
                })

            if better:
                print('New best epoch!')
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))

            if epochs_without_improvement >= args.early_stop_patience:
                print(f'Early stopping triggered after {epochs_without_improvement} epochs with no improvement.')
                break

            losses.append({
                'epoch': epoch, 
                'train': train_loss, 
                'val': val_loss
            })

            store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
        
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_last.pt'))

    # Evaluation functions remain the same
    def run_evaluation(model, data_loader, title):
        print(f'\n===== {title} =====')
        model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))
        map_score, ap_score = evaluate(model, data_loader, nms_window=5)
        
        # Print per-class results
        table = [[name, f"{score*100:.2f}"] for name, score in zip(classes.keys(), ap_score)]
        print(tabulate(table, ["Class", "Average Precision"], tablefmt="grid"))
        
        # Print average results
        avg_table = [
            ["Average 12", f"{map_score*100:.2f}"],
            ["Average 10", f"{np.mean(ap_score[:-2])*100:.2f}"]
        ]
        print(tabulate(avg_table, ["", "Average Precision"], tablefmt="grid"))
        print('=' * 40)

    # Run evaluations
    run_evaluation(model, test_data, "TEST INFERENCE OVER BEST MODEL")
    run_evaluation(model, val_data_video, "VALIDATION INFERENCE OVER BEST MODEL")
    
    # Last model evaluation
    print('\n===== TEST INFERENCE OVER LAST MODEL =====')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_last.pt')))
    map_score, ap_score = evaluate(model, test_data, nms_window=5)
    table = [[name, f"{score*100:.2f}"] for name, score in zip(classes.keys(), ap_score)]
    print(tabulate(table, ["Class", "Average Precision"], tablefmt="grid"))
    print('=' * 40)

    print('\nCORRECTLY FINISHED TRAINING AND INFERENCE')

if __name__ == '__main__':
    main(get_args())