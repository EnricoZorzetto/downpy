#!/bin/bash
# SBATCH -o down_gauges.out
# SBATCH -e down_gauges.err
module load Anaconda3/3.5.2
python download_hpd_from_noaa.py
