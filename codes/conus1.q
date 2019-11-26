#!/bin/bash
module load Anaconda3/3.5.2
#SBATCH -o conus1.out
#SBATCH -e conus1.err
python main_conus_1.py
