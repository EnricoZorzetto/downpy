#!/bin/bash
# SBATCH -o down_tmpa.out
# SBATCH -e down_tmpa.err
module load Anaconda3/3.5.2
python download_tmpa_data.py
