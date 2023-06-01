import argparse
import os

import utils.utils
import utils.batch
import utils.config
import utils.libmain
import models.model

## Cross-validation every subject on SEED dataset.
# @param dirname: dataset directory
# @param regular_str: be used to match right filename
#
def sub_independent_on_SEED(result_dir, dirname, regular_str, cross_validate, class_labels):
    # load data and resort.
    data, labels = utils.utils.load_SEED_independent(dirname, regular_str, class_labels)

    # get result of cross-validation.
    result_list = cross_validate(result_dir, data, labels)

    return result_list


def get_args():
    parser = argparse.ArgumentParser()

    # Experimental settings.
    parser.add_argument('--device', default="cuda:0", help='the device to use')
    parser.add_argument('--max_epoch', type=int, default=200, help='# epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (# nodes)')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument("--patient", type=int, default=8, help="epoch patient")
    
    # Training-params
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate for self-attention model.')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='Initial learning rate for self-attention model.')
     
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    result_dir = "./results"
    dirname = "./data/clipped_data/SEED"
    fold_num = 15
    regular_str = r"\d+_\d+"
    model_class = models.model.Transformer_binary
    split_database_gen = utils.utils.N_cross_split_trial_SEED
    MyDataset_class = utils.batch.MyDataset_SEED
    class_emotions = utils.config.SEED_class_emotions_pos_neg
    class_labels = utils.config.SEED_class_labels_pos_neg

    print(args)

    cross_validate = utils.libmain.gen_cross_validate(
            args,
            model_class,
            fold_num,
            split_database_gen,
            MyDataset_class,
            class_emotions,
            result_dir_prefix = "fold_"
    )
    result_list = sub_independent_on_SEED(result_dir, dirname, regular_str, cross_validate, class_labels)
    print(result_list)









