import argparse
import os
import re
import atexit
import shutil
import statistics
import tensorboardX
import pandas
import seaborn
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import utils.utils
import utils.config
import utils.batch
import models.model
import utils.exception


def del_records():
    run_dir = log_writer.logdir
    checkpoints_dir = os.path.join("checkpoints", experiment_name)
    result_dir = os.path.join("results", experiment_name)
    
    log_writer.close()
    option = input("Delete records of this experiment?(Y/y or others):")
    if option.lower() == 'y':
        shutil.rmtree(run_dir)
        print(f"{run_dir} has been DELETED!")
        if os.path.exists(checkpoints_dir):
            shutil.rmtree(checkpoints_dir)
            print(f"{checkpoints_dir} has been DELETED!")
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
            print(f"{result_dir} has been DELETED!")


def get_all_predictions_and_labels(args, model, dataset):
    device = torch.device(args.device)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    predictions_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            # adjust data to device
            features = features.to(device)
            labels = labels.to(device)

            results, predictions = model(features)
            predictions_list.append(predictions)
            labels_list.append(labels)
    
    all_predictions = torch.cat(predictions_list)
    all_labels = torch.cat(labels_list)
    return all_predictions, all_labels


def get_detailed_accuracy(result_dir, args, model, checkpoint_filepath, dataset, class_emotions):
    model.load_state_dict(torch.load(checkpoint_filepath))
    predictions, labels = get_all_predictions_and_labels(args, model, dataset)
    if args.device != "cpu":
        predictions, labels = predictions.cpu(), labels.cpu()

    conf_matrix = confusion_matrix(labels, predictions)
    conf_matrix_norm = confusion_matrix(labels, predictions, normalize='true')

    # Some class are empty
    if conf_matrix.shape[0] != len(class_emotions):
        class_list = utils.exception.get_class_num(len(class_emotions), predictions)
        for i in range(len(class_emotions)):
            if class_list[i] == 0:
                conf_matrix = utils.exception.insert_zero(i, conf_matrix)
                conf_matrix_norm = utils.exception.insert_zero(i, conf_matrix_norm)

    confmatrix_df = pandas.DataFrame(conf_matrix, index=class_emotions, columns=class_emotions)
    confmatrix_df_norm = pandas.DataFrame(conf_matrix_norm, index=class_emotions, columns=class_emotions)

    plt.figure(figsize=(16, 6))
    seaborn.set_theme(font_scale=1.8) # emotion label and title size
    plt.subplot(1,2,1)
    plt.title('Confusion Matrix')
    seaborn.heatmap(confmatrix_df, annot=True, fmt="d")
    plt.subplot(1,2,2)
    plt.title('Normalized Confusion Matrix')
    seaborn.heatmap(confmatrix_df_norm*100, annot=True, fmt=".2f")
    acc = torch.sum(predictions==labels) / len(labels)

    picture_path = os.path.join(result_dir, "confusion_matrix.jpg")
    plt.savefig(picture_path)
    
    acc_file = os.path.join(result_dir, "log.txt")
    with open(acc_file, 'w') as acf:
        acf.write(f"The accuracy on is {acc} which base on {checkpoint_filepath}.")
    
    return acc


def get_accuracy_loss_on_dataset(args, model, dataset):
    device = torch.device(args.device)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    count = 0
    loss_list = []
    model.eval()
    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            # adjust data to device
            features = features.to(device)
            labels = labels.to(device)

            results, predictions = model(features)
            count += torch.sum(labels==predictions)

            loss = model.loss(results, labels)
            loss_list.append(loss.item())

    dataset_acc = count / len(dataset)
    dataset_loss = statistics.mean(loss_list)
    
    return dataset_acc, dataset_loss


def train(checkpoints_dir, log_writer, args, model, dataset_train, dataset_val):
    device = torch.device(args.device)
    dataloader = DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # in training
    best_val_acc = 0.0
    patient = 0
    best_checkpoint_filepath = ""

    for epoch in range(args.max_epoch):
        count = 0
        loss_list = []
        model.train()
        print(f"Epoch: {epoch}")
        for features, labels in tqdm(dataloader):
            # adjust data to device
            features = features.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            results, predictions = model(features)
            loss = model.loss(results, labels)
            loss.backward()
            opt.step()

            count += torch.sum(labels==predictions)
            loss_list.append(loss.item())
        
        epoch_acc = count / len(dataset_train)
        epoch_loss = statistics.mean(loss_list)
        val_acc, val_loss = get_accuracy_loss_on_dataset(args, model, dataset_val)

        log_writer.add_scalars("acc", {"Epoch":epoch_acc, "Val":val_acc}, epoch)
        log_writer.add_scalars("loss", {"Epoch":epoch_loss, "Val":val_loss}, epoch)
        print(f"Epoch acc = {epoch_acc}, Epoch loss = {epoch_loss}")
        print(f" Val acc = {val_acc}, Val loss = {val_loss}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_filepath = os.path.join(checkpoints_dir, f"epoch{epoch}.pt")
            torch.save(model.state_dict(), best_checkpoint_filepath)
            patient = 0
        else:
            patient += 1

        if patient == args.patient:
            break

    return best_checkpoint_filepath


def get_args():
    parser = argparse.ArgumentParser()

    # Experimental settings.
    parser.add_argument('--device', default="cuda:0", help='the device to use')
    parser.add_argument('--max_epoch', type=int, default=200, help='# epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (# nodes)')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument("--patient", type=int, default=16, help="epoch patient")
    
    # Training-params
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate for self-attention model.')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='Initial learning rate for self-attention model.')
     
    args = parser.parse_args()
    return args


def gen_cross_validate(args, model_class, fold_num, sample_length, window_length, result_dir_prefix = "fold_"):
    def cross_validate(result_rootdir, data, labels):
        acc_list = []
        gen_10_fold_database = utils.utils.N_cross_split_trial_DEAP(data, labels, fold_num)
        for fold in range(fold_num):
            data_train, labels_train, data_val, labels_val = gen_10_fold_database.__next__()
            train_database = utils.batch.MyDataset_DEAP(data_train, labels_train, sample_length, window_length)
            val_database = utils.batch.MyDataset_DEAP(data_val, labels_val, sample_length, window_length)

            exp_dir = os.path.join(result_rootdir, result_dir_prefix + str(fold))
            checkpoints_dir = os.path.join(exp_dir, "checkpoints")
            os.makedirs(checkpoints_dir)
            log_writer = tensorboardX.SummaryWriter(logdir=os.path.join(exp_dir, "runs"))
            acc_dir = os.path.join(exp_dir, "results" )
            os.makedirs(acc_dir)

            model = model_class().to(torch.device(args.device))
            best_checkpoint_filepath = train(checkpoints_dir, log_writer, args, model, train_database, val_database)
            acc = get_detailed_accuracy(acc_dir, args, model, best_checkpoint_filepath, val_database, utils.config.DEAP_class_emotions)
            acc_list.append(acc)
        
        # display and save information.
        log_file = os.path.join(result_rootdir, "log.txt")
        with open(log_file, 'w') as log:
            log.write(str(acc_list))
        
        return acc_list
    return cross_validate


def regular_file_list(original_file_list, regular):
    new_list = []
    for filename in original_file_list:
        if re.match(regular, filename):
            new_list.append(filename)

    return new_list


## Cross-validation every subject on DEAP dataset.
# @param dirname: dataset directory
#
def valid_all(result_dir, dirname, regular_str, cross_validate, result_prefix="sub_"):
    tmp_file_list = os.listdir(dirname)
    file_list = regular_file_list(tmp_file_list, regular_str)
    
    print(file_list)

    result_list = []
    for filename in file_list:
        sub_dir = os.path.join(result_dir, result_prefix + str(filename))
        filepath = os.path.join(dirname, filename)

        # load data and resort
        data, labels = utils.utils.load_DEAP(filepath, class_number)
        new_order = utils.utils.get_order_to_cross(labels)
        data, labels = data[new_order][...], labels[new_order]

        acc_list = cross_validate(sub_dir, data, labels)
        result_list.append(acc_list)

        print(acc_list)

    # save information.
    log_file = os.path.join(result_dir, "log.txt")
    with open(log_file, 'w') as log:
        log.write(str(result_list))
    
    return result_list


## generate cross-validation function base on SEED dataset.
#
def gen_cross_validate_on_SEED(args, model_class, fold_num, result_dir_prefix = "fold_"):
    def cross_validate(result_rootdir, data, labels):
        acc_list = []
        gen_10_fold_database = utils.utils.N_cross_split_trial_SEED(data, labels, fold_num)
        for fold in range(fold_num):
            # get database of train, val.
            data_train, labels_train, data_val, labels_val = gen_10_fold_database.__next__()
            train_database = utils.batch.MyDataset_SEED(data_train, labels_train)
            val_database = utils.batch.MyDataset_SEED(data_val, labels_val)

            # point to derectory to save generated data.
            exp_dir = os.path.join(result_rootdir, result_dir_prefix + str(fold))
            checkpoints_dir = os.path.join(exp_dir, "checkpoints")
            os.makedirs(checkpoints_dir)
            log_writer = tensorboardX.SummaryWriter(logdir=os.path.join(exp_dir, "runs"))
            acc_dir = os.path.join(exp_dir, "results" )
            os.makedirs(acc_dir)

            # train and get result.
            model = model_class().to(torch.device(args.device))
            best_checkpoint_filepath = train(checkpoints_dir, log_writer, args, model, train_database, val_database)
            acc = get_detailed_accuracy(acc_dir, args, model, best_checkpoint_filepath, val_database, utils.config.SEED_class_emotions)
            acc_list.append(acc)

        # display and save information.
        log_file = os.path.join(result_rootdir, "log.txt")
        with open(log_file, 'w') as log:
            log.write(str(acc_list))

        return acc_list
    return cross_validate


## Cross-validation every subject on SEED dataset.
# @param dirname: dataset directory
# @param regular_str: be used to match right filename
#
def valid_all_on_SEED(result_dir, dirname, regular_str, cross_validate, result_prefix="sub_"):
    # get right file list
    tmp_file_list = os.listdir(dirname)
    file_list = regular_file_list(tmp_file_list, regular_str)

    print(file_list)

    result_list = []
    for filename in file_list:
        # point to result direstory and source file.
        sub_dir = os.path.join(result_dir, result_prefix + str(filename))
        os.makedirs(sub_dir)

        # load data and resort.
        filepath = os.path.join(dirname, filename)
        data, labels = utils.utils.load_SEED(filepath)

        # get result of cross-validation.
        acc_list = cross_validate(sub_dir, data, labels)
        result_list.append(acc_list)

        print(acc_list)

    # save information.
    log_file = os.path.join(result_dir, "log.txt")
    with open(log_file, 'w') as log:
        log.write(str(result_list))

    return result_list


def main():
    args = get_args()
    print(args)
    print("Experiment name:", experiment_name)

    device = torch.device(args.device)

    if args.dataset_type == "DEAP":
        train_x, train_y, val_x, val_y, test_x, test_y = \
                utils.utils.get_dataset_train_val_test_from_DEAP(args.dataset_path)
        
        dataset_train = utils.batch.MyDataset_DEAP(train_x, train_y)
        dataset_val = utils.batch.MyDataset_DEAP(val_x, val_y)
        dataset_test = utils.batch.MyDataset_DEAP(test_x, test_y)

        model = models.model.Transformer_DEAP().to(device)

    else:
        if args.dataset_type == "SEED":
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
                    utils.utils.get_dataset_train_val_test(args.dataset_path)
            classes = 3
        elif args.dataset_type == "SEED_IV":
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
                    utils.utils.get_dataset_train_val_test_from_SEED_IV(args.dataset_path)
            classes = 4
        else:
            raise Exception("Error about dataset_type")

        dataset_train = utils.batch.MyDataset(train_paths, train_labels)
        dataset_val = utils.batch.MyDataset(val_paths, val_labels)
        dataset_test = utils.batch.MyDataset(test_paths, test_labels)
        
        model = models.model.Transformer(classes).to(device)
    
    print("dataset_train:", len(dataset_train))
    print("dataset_val:", len(dataset_val))
    print("dataset_test:", len(dataset_test))
    
    if args.train:
        log_writer = tensorboardX.SummaryWriter(logdir=os.path.join("runs", experiment_name))
        atexit.register(del_records)
        
        best_checkpoint_filepath = train(args, model, dataset_train, dataset_val)
        model.load_state_dict(torch.load(best_checkpoint_filepath))
        acc, result_path = get_detailed_accuracy(args, model, dataset_test)
        print(f"The test accuracy on best val is {acc}, and its chechpoint was saved in {best_checkpoint_filepath}.")
        
        acc_file = os.path.join("results", experiment_name, "test_acc.txt")
        with open(acc_file, 'w') as acf:
            acf.write(f"The test accuracy on best val is {acc}, and its chechpoint was saved in {best_checkpoint_filepath}.")
        #plt.show()
        plt.clf()
    
    elif args.pretrained_train:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        
        log_writer = tensorboardX.SummaryWriter(logdir=os.path.join("runs", experiment_name))
        atexit.register(del_records)

        best_checkpoint_filepath = train(args, model, dataset_train, dataset_val)
        model.load_state_dict(torch.load(best_checkpoint_filepath))
        acc, result_path = get_detailed_accuracy(args, model, dataset_test)
        print(f"The test accuracy on best val is {acc}, and its checkpoint was saved in {best_checkpoint_filepath}.")

        acc_file = os.path.join("results", experiment_name, "test_acc.txt")
        with open(acc_file, 'w') as acf:
            acf.write(f"The test accuracy on best val is {acc}, and its checkpoint was saved in {best_checkpoint_filepath}.")
        #plt.show()
        plt.clf()
    
    elif args.test:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))

        acc, _ = get_accuracy_loss_on_dataset(args, model, dataset_test)
        print(f"The test acc = {acc}")

        result_dir = os.path.join("results", experiment_name)
        os.makedirs(result_dir)
        acc_file = os.path.join(result_dir, "test_acc.txt")
        with open(acc_file, 'w') as acf:
            acf.write(f"The test acc = {acc} on {args.checkpoint}.")
    
    elif args.detailed_test:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))

        acc, result_path = get_detailed_accuracy(args, model, dataset_test)
        print(f"The test accuracy is {acc}, and its result was saved in {result_path}.")

        acc_file = os.path.join("results", experiment_name, "test_acc.txt")
        with open(acc_file, 'w') as acf:
            acf.write(f"The test acc = {acc} on {args.checkpoint}.")
        #plt.show()
        plt.clf()
    
    else:
        pass


def valid_all_on_DEAP():
    args = get_args()

    result_dir = "./result"
    dirname = "./data/row_data/DEAP"
    fold_num = 10
    sample_length = 128
    window_length = 32
    class_number = 0
    regular_str = r"s\d+\.mat"

    print(args)

    cross_validate = gen_cross_validate(args, models.model.Transformer_DEAP, fold_num, sample_length, window_length, result_dir_prefix = "fold_")
    result_list = valid_all(result_dir, dirname, regular_str, cross_validate, result_prefix="sub_")
    print(result_list)

if __name__ == "__main__":
    args = get_args()

    result_dir = "./result"
    dirname = "./data/clipped_data/SEED"
    fold_num = 5
    regular_str = r"\d+_\d+"
    model_class = models.model.Transformer_12_2

    print(args)

    cross_validate = gen_cross_validate_on_SEED(args, model_class, fold_num, result_dir_prefix = "fold_")
    result_list = valid_all_on_SEED(result_dir, dirname, regular_str, cross_validate, result_prefix="sub_")
    print(result_list)









