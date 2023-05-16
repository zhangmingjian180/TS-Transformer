import argparse
import scipy.io
import os
import shutil
import pickle
import numpy as np

import utils.config

from tqdm import tqdm
from utils.preprocess import split_matrix, get_sorted_keys


def add_dir(info_list, dirname):
    for i in range(len(info_list)):
        info_list[i]["path"] = os.path.join(dirname, info_list[i]["path"])


def notmalize_array(array):
    """
    for i in range(array.shape[0]):
        array[i][...] = (array[i] - array[i].mean()) / array.std()
    """
    array = np.tanh(array / 100)

    return array


def output_array_to_dir(destination_dir, source_matrix, label, win_length, win_move):
    # split matrix
    matrix_list = split_matrix(source_matrix, win_length, win_move)

    # transfor and save graph, record in information list
    info_list = []
    for i in range(len(matrix_list)):
        array = notmalize_array(matrix_list[i])
        filename = str(i) + ".pkl"
        save_path = os.path.join(destination_dir, filename)
        
        with open(save_path, "wb") as f:
            pickle.dump(array, f)
        
        info_list.append({"label":label, "path":filename})
    
    # save information list
    save_path = os.path.join(destination_dir, "info_list.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(info_list, f)

    return info_list


def output_file_to_dir(destination_dir, source_file_path, labels, win_length, win_move):
    mat_dict = scipy.io.loadmat(source_file_path)
    varname_list = get_sorted_keys(mat_dict.keys())

    info_list = []
    for idx, varname in tqdm(enumerate(varname_list)):
        corresponding_dir = os.path.join(destination_dir, varname)
        os.makedirs(corresponding_dir)

        tmp_info_list = output_array_to_dir(corresponding_dir, mat_dict[varname], labels[idx], win_length, win_move)
        
        add_dir(tmp_info_list, varname)
        info_list.extend(tmp_info_list)

    save_path = os.path.join(destination_dir, "info_list.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(info_list, f)

    return info_list


## This function is aimed to pre-process EEG data of source dir.
# @param destination_dir  It must exists and be empty.
# @param source_dir  It include one or more .mat file.
# @param labels The labels in every .mat file.
#
def output_dir_to_dir(destination_dir, source_dir, labels, win_length, win_move):
    info_list = []
    for filename in os.listdir(source_dir):
        dirname = filename.split(".")[0]
        origin_file_path = os.path.join(source_dir, filename)
        corresponding_dir = os.path.join(destination_dir, dirname)
        os.makedirs(corresponding_dir)

        tmp_info_list = output_file_to_dir(corresponding_dir, origin_file_path, labels, win_length, win_move)
        
        add_dir(tmp_info_list, dirname)
        info_list.extend(tmp_info_list)

    save_path = os.path.join(destination_dir, "info_list.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(info_list, f)

    return info_list


## This function is aimed to pre-process SEED_IV dataset.
# @param destination_dir  It must exists and be empty.
# @param source_dir  It is path of SEED_IV dataset.
# @param labels_dict  The labels of whole SEED_IV.
#
def process_SEED_IV(destination_dir, source_dir, labels_dict, win_length, win_move):
    assert (os.listdir(destination_dir) == [] and os.listdir(destination_dir) == []), f"{destination_dir} must exists and be empty."

    info_list = []
    for dirname in os.listdir(source_dir):
        labels = labels_dict[dirname]

        origin_dir = os.path.join(source_dir, dirname)
        corresponding_dir = os.path.join(destination_dir, dirname)
        os.makedirs(corresponding_dir)

        tmp_info_list = output_dir_to_dir(corresponding_dir, origin_dir, labels, win_length, win_move)

        add_dir(tmp_info_list, dirname)
        info_list.extend(tmp_info_list)

    save_path = os.path.join(destination_dir, "info_list.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(info_list, f)


## This function is aimed to pre-process SEED dataset.
# @param destination_dir  It must exists and be empty.
# @param source_dir  It is path of SEED dataset.
# @param labels The labels in every .mat file.
#
def process_SEED(destination_dir, source_dir, labels, win_length, win_move):
    assert (os.path.exists(destination_dir) and os.listdir(destination_dir) == []), f"{destination_dir} must exists and be empty."
    output_dir_to_dir(destination_dir, source_dir, labels, win_length, win_move)


## This file is used to pre-process EEG dataset include SEED, SEED_IV.
#  It will afford a logical interface for raw dataset.
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--destination_dir", default="./data/clipped_data/SEED")
    parser.add_argument("--source_dir", default="./data/row_data/SEED")
    parser.add_argument("--win_length", type=int, default=400, help="the length of channel")
    parser.add_argument("--win_move", type=int, default=80, help="the length of window moving")

    args = parser.parse_args()
    print(args)

    dataset = os.path.split(args.source_dir)[-1]

    if dataset == "SEED":
        process_SEED(args.destination_dir, args.source_dir, utils.config.labels, args.win_length, args.win_move)
    elif dataset == "SEED_IV":
        process_SEED_IV(args.destination_dir, args.source_dir, utils.config.SEED_IV_labels, args.win_length, args.win_move)
    else:
        raise Exception("Error about dataset")
