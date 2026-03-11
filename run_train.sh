#!/bin/bash
source /home/binzhou/miniconda/etc/profile.d/conda.sh
conda activate CAVTL
cd /home/binzhou/Documents/route_TSC
python start.py --algo alpha_router 2>&1 | tee train_alpha_router.log
