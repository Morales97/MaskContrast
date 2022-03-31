#!/bin/bash
#
#SBATCH --job-name=seg_test
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=16:00:00

#python main.py --expt_name=test --config_env configs/env.yml --config_exp configs/VOCSegmentation_supervised_saliency_model.yml
#python main_single_thread.py --expt_name=test_no_acc --config_env configs/env.yml --config_exp configs/VOCSegmentation_supervised_saliency_model.yml

# cityscapes deeplab
#python main_single_thread.py --expt_name=test --config_env configs/env.yml --config_exp configs/cityscapes.yml

#cityscapes+GTA deeplab
#python main_single_thread.py --expt_name=CS_GTA_mn_deeplab --config_env configs/env.yml --config_exp configs/cityscapes_gta_deeplab.yml


#cityscapes lraspp
#python main_single_thread.py --expt_name=CS_mn_lraspp_600_epochs --config_env configs/env.yml --config_exp configs/cityscapes_lraspp.yml

#cityscapes+GTA lraspp
#python main_single_thread.py --expt_name=CS_GTA_mn_lraspp_600_epochs --config_env configs/env.yml --config_exp configs/cityscapes_gta_lraspp.yml

# ---- with supervised object masks -----
#cityscapes lraspp
#python main_single_thread.py --expt_name=CS_100sup_mn_lraspp_all_classes --config_env configs/env.yml --config_exp configs/cityscapes_lraspp.yml

# CS + GTA lraspp
python main_single_thread.py --expt_name=CS_GTA_sup_masks --config_env configs/env.yml --config_exp configs/cityscapes_gta_lraspp.yml
