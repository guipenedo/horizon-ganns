#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00
#SBATCH -n 6
#SBATCH --begin=now
#SBATCH --mail-user=REDACTED
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=o_cgan_gp
#SBATCH -o /scratch/students/g.cabral/results/o_cgan_gp.%j.out # STDOUT
#SBATCH -e /scratch/students/g.cabral/results/o_cgan_gp.%j.err # STDERR

module load python/3.8
cd /scratch/students/g.cabral/
source activate torch
echo "Checking if dataset is in /fasttmp..."

DIR="/fasttmp/g.cabral/observations_5s"
if ! [ -d "$DIR" ]; then
  echo "Copying dataset to /fasttmp..."
  cp -r /scratch/students/g.cabral/observations_5s /fasttmp/g.cabral/observations_5s
  echo "Finished copying"
fi
echo "Starting script execution"
cd code
python optuna_cgan_gp.py
