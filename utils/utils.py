import pickle
import os
import numpy as np
import queue
import re


import scipy.io
from sklearn.model_selection import train_test_split

import utils.config

from utils.preprocess import get_sorted_keys


def regular_file_list(original_file_list, regular):
    new_list = []
    for filename in original_file_list:
        if re.match(regular, filename):
            new_list.append(filename)

    return new_list


def get_order_to_cross(labels):
    new_order = []
    class_num = int(max(labels)) + 1
    q_list = []
    for i in range(class_num):
        q_list.append(queue.Queue())

    for i in range(len(labels)):
        q_list[int(labels[i])].put(i)

    while len(new_order) < len(labels):
        for q in q_list:
            if not q.empty():
                new_order.append(q.get())
    
    return new_order


def load_pickle(path):
    with open(path, "rb") as f:
        result = pickle.load(f)
    return result


def normalize_DEAP(x):
    return (x - x.mean()) / x.std()


## This function is used to load single subject.
# @param root: root directory which include 15 sub dir.
# @param class_labels: point to labels to classify (default -1, 0, 1).
# @return data_paths: shape(15, N)
# @return labels: shape(15, N)
#
def load_SEED(root, class_labels=utils.config.SEED_labels):
    all_dir_list = get_sorted_keys(os.listdir(root))

    data_paths = []
    labels = []
    for i in range(15):
        session_paths = []
        session_labels = []
        
        dir_name = os.path.join(root, all_dir_list[i])
        info_file = os.path.join(dir_name, "info_list.pkl")
        info_list = load_pickle(info_file)
        for info in info_list:
            session_paths.append(os.path.join(dir_name, info["path"]))
            session_labels.append(class_labels.index(info["label"]))
        
        data_paths.append(session_paths)
        labels.append(session_labels)
    
    return data_paths, labels


def get_paths_and_labels(root, position_number, class_labels):
    all_dir_list = get_sorted_keys(os.listdir(root))
    
    x_paths = []
    labels = []
    for logic_i in position_number:
        i = logic_i - 1
        dir_name = os.path.join(root, all_dir_list[i])
        info_file = os.path.join(dir_name, "info_list.pkl")
        info_list = load_pickle(info_file)
        
        for info in info_list:
            x_paths.append(os.path.join(dir_name, info["path"]))
            labels.append(class_labels.index(info["label"]))

    return x_paths, labels


def train_val_test_split(data, labels, test_size=0.2, val_size=0.2):
    data_train_t, data_test, labels_train_t, labels_test = train_test_split(data, labels, test_size=test_size)
    data_train, data_val, labels_train, labels_val = train_test_split(data_train_t, labels_train_t, test_size=val_size)
    
    return data_train, labels_train, data_val, labels_val, data_test, labels_test


def split_data(original_data, original_labels, sample_length=256):
    sample_num = original_data.shape[0] * original_data.shape[2] // sample_length
    x = original_data.transpose(1, 0, 2).reshape(original_data.shape[1], sample_num, sample_length).transpose(1, 0, 2)
    y = original_labels.repeat(original_data.shape[2] // sample_length, axis=0)

    return x, y


def load_DEAP(filename, class_number):
    eeg_dict = scipy.io.loadmat(filename)
    data = eeg_dict["data"][:, :32, 128*3:].astype(np.float32)  # (40, 32, 7680)
    labels = np.trunc(eeg_dict["labels"][:, class_number].astype(np.float32) / 5)  # (40)
    return data, labels


def N_cross_split_trial_DEAP(data, labels, fold_num):
    samples_num = data.shape[0]
    test_samples_num = samples_num // fold_num
    train_samples_num = samples_num - test_samples_num
    for i in range(fold_num):
        test_start_point = i * test_samples_num
        train_start_point = test_start_point + test_samples_num
        test_list = list(range(test_start_point, train_start_point))
        train_list = list(range(0, test_start_point)) + list(range(train_start_point, samples_num))
        
        data_train, data_test = data[train_list][...], data[test_list][...]
        labels_train, labels_test = labels[train_list], labels[test_list]

        yield data_train, labels_train, data_test, labels_test
    return

## The function is aimed to generate a new list from old list based on position list.
def get_new_list(source_list, positional_number):
    new_list = []
    for i in positional_number:
        new_list.append(source_list[i])
    return new_list


## This function is used to return a generator to genarate train and val data.
# @param data: Its format is [[], [], ...]
# @param labels: Its format is [[], [], ...]
# @return: generator
#
def N_cross_split_trial_SEED(data, labels, fold_num):
    samples_num = len(data)
    test_samples_num = samples_num // fold_num
    train_samples_num = samples_num - test_samples_num
    for i in range(fold_num):
        test_start_point = i * test_samples_num
        train_start_point = test_start_point + test_samples_num
        test_list = list(range(test_start_point, train_start_point))
        train_list = list(range(0, test_start_point)) + list(range(train_start_point, samples_num))

        data_train, data_test = get_new_list(data, train_list), get_new_list(data, test_list)
        labels_train, labels_test = get_new_list(labels, train_list), get_new_list(labels, test_list)

        yield data_train, labels_train, data_test, labels_test
    return


def get_dataset_train_val_test_from_DEAP(filename, class_number=0):
    eeg_dict = scipy.io.loadmat(filename)
    data = np.tanh(eeg_dict["data"][:, :32, 128*3:].astype(np.float32) / 25)  # (40, 32, 7680)
    labels = np.trunc(eeg_dict["labels"][:, class_number].astype(np.float32) / 5)  # (40)

    data_train, labels_train, data_val, labels_val, data_test, labels_test = train_val_test_split(data, labels)

    train_x, train_y = split_data(data_train, labels_train)
    val_x, val_y = split_data(data_val, labels_val)
    test_x, test_y = split_data(data_test, labels_test)

    return train_x, train_y, val_x, val_y, test_x, test_y


def get_dataset_train_val_test(root):
    train_paths, train_labels = get_paths_and_labels(root, utils.config.dataset_partition["train"], utils.config.class_labels)
    val_paths, val_labels = get_paths_and_labels(root, utils.config.dataset_partition["val"], utils.config.class_labels)
    test_paths, test_labels = get_paths_and_labels(root, utils.config.dataset_partition["test"], utils.config.class_labels)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_dataset_train_val_test_from_SEED_IV(root):
    session = os.path.split(os.path.split(root)[0])[1]
    if session == "1":
        dataset_partition = utils.config.SEED_IV_session1_dataset_partition
    elif session == "2":
        dataset_partition = utils.config.SEED_IV_session2_dataset_partition
    elif session == "3":
        dataset_partition = utils.config.SEED_IV_session3_dataset_partition
    else:
        raise Exception("Error about getting dataset.")
    
    train_paths, train_labels = get_paths_and_labels(root, dataset_partition["train"], utils.config.SEED_IV_class_labels)
    val_paths, val_labels = get_paths_and_labels(root, dataset_partition["val"], utils.config.SEED_IV_class_labels)
    test_paths, test_labels = get_paths_and_labels(root, dataset_partition["test"], utils.config.SEED_IV_class_labels)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def classify(all_dir_list):
    # build empty dict.
    result_dict = {}
    for i in range(1, 16):
        result_dict[i] = []
    for e in all_dir_list:
        i = int(e.split("_")[0])
        result_dict[i].append(e)
    return result_dict


# The has been passed test.
def load_SEED_independent(dirname, regular_str, class_labels=utils.config.SEED_labels):
    tmp_file_list = os.listdir(dirname)
    all_dir_list = regular_file_list(tmp_file_list, regular_str)

    result_dict = classify(all_dir_list)

    data_paths = [[]] * 15
    labels = [[]] * 15

    for i in range(1, 16):
        for j in range(3):
            session_paths = []
            session_labels = []

            dir_name = os.path.join(dirname, result_dict[i][j])
            info_file = os.path.join(dir_name, "info_list.pkl")
            info_list = load_pickle(info_file)
            for info in info_list:
                if info["label"] in class_labels:
                    session_paths.append(os.path.join(dir_name, info["path"]))
                    session_labels.append(class_labels.index(info["label"]))

            data_paths[i-1] = data_paths[i-1] + session_paths
            labels[i-1] = labels[i-1] + session_labels

    return data_paths, labels





