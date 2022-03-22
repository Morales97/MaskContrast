#!/bin/bash
#
#SBATCH --job-name=seg_test
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main.py --expt_name=test --config_env configs/env.yml --config_exp configs/VOCSegmentation_supervised_saliency_model.yml
#python main_single_thread.py --expt_name=test_no_acc --config_env configs/env.yml --config_exp configs/VOCSegmentation_supervised_saliency_model.yml

# cityscapes
python main_single_thread.py --expt_name=test_CS_no_acc --config_env configs/env.yml --config_exp configs/cityscapes.yml

