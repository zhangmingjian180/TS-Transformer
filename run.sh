#!/bin/bash


###--- DEAP ---###
# subjet-dependent experiment.
# cross-validation
python sub_dependent_DEAP.py


###--- SEED ---###
# subjet-dependent experiment.
# process data
python preprocess_SEED_series.py --win_move=80
# cross-validation
python sub_dependent_SEED.py

###--- SEED ---###
# subjet-independent experiment.
# process data
python preprocess_SEED_series.py --win_move=320
# cross-validation
python sub_independent_SEED.py

###--- SEED ---###
# subjet-independent experiment on positive and negtive emotion.
# process data
python preprocess_SEED_series.py --win_move=320
# cross-validation
python sub_independent_SEED_pos_neg.py
