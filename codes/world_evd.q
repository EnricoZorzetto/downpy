#!/bin/bash

#SBATCH -o world.out
#SBATCH -e world.err
#SBATCH --mem=16G
module load Anaconda3/3.5.2
python main_evd_maps_from_raw_yearly_old_backup.py
