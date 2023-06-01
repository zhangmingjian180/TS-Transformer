import statistics
import tensorboardX
import pandas
import seaborn
import os
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import utils.utils
import utils.exception

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

## generate cross-validation function.
#
def gen_cross_validate(args, model_class, fold_num, split_database_gen, MyDataset_class, class_emotions, result_dir_prefix = "fold_"):
    def cross_validate(result_rootdir, data, labels):
        acc_list = []
        database_gen = split_database_gen(data, labels, fold_num)
        for fold in range(fold_num):
            """
            if fold not in [5, 11]:
                continue
            """
            # get database of train, val.
            data_train, labels_train, data_val, labels_val = database_gen.__next__()
            train_database = MyDataset_class(data_train, labels_train)
            val_database = MyDataset_class(data_val, labels_val)

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
            acc = get_detailed_accuracy(acc_dir, args, model, best_checkpoint_filepath, val_database, class_emotions)
            acc_list.append(acc)

        # display and save information.
        log_file = os.path.join(result_rootdir, "log.txt")
        with open(log_file, 'w') as log:
            log.write(str(acc_list))

        return acc_list
    return cross_validate

