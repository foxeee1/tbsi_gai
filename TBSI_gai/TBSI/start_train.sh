#!/usr/bin/env bash
source /root/autodl-tmp/conda_envs/tbsi/etc/profile.d/conda.sh
conda activate tbsi
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/TBSI_gai/TBSI
exec python tracking/train.py \
    --script tbsi_track \
    --config vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce \
    --save_dir ./output \
    --mode single \
    >> output/logs/train_tbsi_track-vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce.log 2>&1
