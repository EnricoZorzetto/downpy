#!/bin/bash
module load Anaconda3/3.5.2
#SBATCH -o conus2.out
#SBATCH -e conus2.err
python main_conus_2.py
