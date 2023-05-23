#!/bin/bash


###--- DEAP ---###
# cross-validation
python main_DEAP.py


###--- SEED ---###
# process data
python preprocess_SEED_series.py
# cross-validation
python main_SEED.py
