#!/bin/bash


###--- DEAP ---###
# train
python main.py --train --dataset_type=DEAP --dataset_path=data/row_data/DEAP/s01.mat --device=cpu --num_workers=2

###--- SEED ---###
# process data
python data_process.py --source_dir=./data/row_data/SEED --destination_dir=./data/clipped_data/SEED
# main
python main.py --train --dataset_type=SEED --dataset_path=./data/clipped_data/SEED/8_20140514


###--- SEED_IV ---###
# process data
python data_process.py
# main train
python main.py --train --dataset_path=./data/clipped_data/SEED_IV/1/2_20150915

# main detailed test
python main.py --detailed_test --checkpoint=./checkpoints/time_2022_12_31_23_59_01_307738/epoch15.pt  --dataset_path=./data/clipped_data/SEED_IV/1/2_20150915


