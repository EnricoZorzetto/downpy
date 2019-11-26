#!/bin/bash

#SBATCH -o world.out
#SBATCH -e world.err
module load Anaconda3/3.5.2
python main_0_extract_daily_tmpa_world.py
