#!/bin/bash
source /home/binzhou/miniconda/etc/profile.d/conda.sh
conda activate CAVTL
cd /home/binzhou/Documents/route_TSC
tensorboard --logdir=./runs --port=6006 --bind_all
