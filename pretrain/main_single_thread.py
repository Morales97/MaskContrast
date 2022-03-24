#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import builtins
import os
import sys
from termcolor import colored

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from data.dataloaders.dataset import DatasetKeyQuery

from modules.moco.builder import ContrastiveModel

from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_train_dataloader, get_optimizer, adjust_learning_rate

from utils.train_utils import train, train_two_datasets
from utils.logger import Logger
from utils.collate import collate_custom
import wandb
import pdb


# Parser
parser = argparse.ArgumentParser(description='Main function')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--nvidia-apex', action='store_true',
                    help='Use mixed precision')

# Distributed params
parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                            help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

# wandb
parser.add_argument('--save_dir', type=str, default='outputs/tmp_last',
                    help='dir to save experiment results to')
parser.add_argument('--project', type=str, default='MaskContrast',
                    help='wandb project to use')
parser.add_argument('--entity', type=str, default='morales97',
                    help='wandb entity to use')
parser.add_argument('--expt_name', type=str, default='',
                    help='Name of the experiment for wandb')

def main():
    args = parser.parse_args()
    args.multiprocessing_distributed = False
    ngpus_per_node = torch.cuda.device_count()

    wandb.init(name=args.expt_name, dir=args.save_dir, config=args, reinit=True, project=args.project, entity=args.entity)
    os.makedirs(args.save_dir, exist_ok=True)

    main_worker(0, ngpus_per_node, wandb, args=args)
    wandb.join()

def main_worker(gpu, ngpus_per_node, wandb, args):
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)

    args.gpu = torch.cuda.current_device()
    p['gpu'] = torch.cuda.current_device()
    p['distributed'] = False

    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = ContrastiveModel(p)
    model.cuda()
    
    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model.parameters())
    print(optimizer)
    amp = None
    
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    #p['train_batch_size'] = int(p['train_batch_size'] / ngpus_per_node)
    #p['num_workers'] = int((p['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)
    
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    train_transform = get_train_transformations()
    print(train_transform)
    train_dataset = DatasetKeyQuery(get_train_dataset(p, transform=None), 
                                    train_transform, 
                                    downsample_sal=not p['model_kwargs']['upsample'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=p['train_batch_size'], shuffle=True, num_workers=p['num_workers'])
    print(colored('Train samples %d for %s' % (len(train_dataset), p['train_db_name']), 'yellow'))
    print(colored(train_dataset, 'yellow'))

    if p['train_db2_name'] is not None:
        train_dataset_2 = DatasetKeyQuery(get_train_dataset(p, transform=None, dataset=p['train_db2_name']), 
                                        train_transform, 
                                        downsample_sal=not p['model_kwargs']['upsample'])
        train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=p['train_batch_size'], shuffle=True, num_workers=p['num_workers'])
        print(colored('Train samples %d for %s' % (len(train_dataset_2), p['train_db2_name']), 'yellow'))
        print(colored(train_dataset_2, 'yellow'))

    # Resume from checkpoint
    if p['load_checkpoint'] and os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(p['checkpoint'], map_location=loc)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        if args.nvidia_apex:
            amp.load_state_dict(checkpoint['amp'])
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Main loop
    print(colored('Starting main loop', 'blue'))
    
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        # Use one dataset
        if p['train_db2_name'] is None:
            eval_train = train(p, train_dataloader, model, optimizer, epoch, amp, wandb)
        # Use two datasets
        else:
            eval_train = train_two_datasets(p, train_dataloader, train_dataloader_2, model, optimizer, epoch, amp, wandb)

        # Checkpoint
        print('Checkpoint ...')
        if args.nvidia_apex:
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'amp': amp.state_dict(), 'epoch': epoch + 1}, 
                        p['checkpoint'])

        else:
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch + 1}, 
                        p['checkpoint'])

        model_artifact = wandb.Artifact('checkpoint_{}'.format(epoch), type='model')
        model_artifact.add_file(p['checkpoint'])
        wandb.log_artifact(model_artifact)

if __name__ == "__main__":
    main()
