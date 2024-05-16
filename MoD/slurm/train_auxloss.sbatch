#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=train1
#SBATCH --output=aux_loss_E10_RR_10_143_val_data_140M%j.out
#SBATCH --error=aux_loss_E10_RR_10_143_val_data_140M%j.err
#SBATCH --time=1-22:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=sdm8499@nyu.edu # send me mail

singularity exec --nv --bind /scratch/sdm8499 --overlay /scratch/sdm8499/pytorchnew/torchgeo_usage.ext3:ro  /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/sdm8499/sushi/MoD_simran_2/MoD_training.py --epochs 10 --dim 1024 --aux_loss True --model_dir /scratch/sdm8499/sushi/MoD_simran_2/model_aux_loss_E10_RR_10_143_val_data_140M --log_path /scratch/sdm8499/sushi/MoD_simran_2/log_aux_loss_E10_RR_10_143_val_data_140M --log_path_debug output_aux_loss_E10_RR_10_143_val_data_140M.log --routing_decision_path_dir /scratch/sdm8499/sushi/MoD_simran_2/paths_aux_loss_E10_RR_10_143_val_data_140M/ "
