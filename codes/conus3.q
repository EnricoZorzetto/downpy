#!/bin/bash
module load Anaconda3/3.5.2
#SBATCH -o conus3.out
#SBATCH -e conus3.err
python main_conus_3.py
