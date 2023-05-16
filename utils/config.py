labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

dataset_partition = {
        "train":[1, 2, 3, 4, 5, 6, 7, 8, 9],
        "val":[10, 11, 12],
        "test":[13, 14, 15]
}

SEED_class_emotions = ["postive", "nature", "negtive"]
class_labels = [1, 0, -1]

SEED_IV_labels = {
        "1":[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        "2":[2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        "3":[1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
}

"""
SEED_IV_session1_dataset_partition = {
        "train":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 19, 20],
        "val":[13, 17, 21, 22],
        "test":[14, 18, 23, 24]
}
"""
SEED_IV_session1_dataset_partition = {
        "train":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 19, 20],
        "test":[13, 17, 21, 22],
        "val":[14, 18, 23, 24]
}
SEED_IV_session2_dataset_partition = {
        "train":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18],
        "val":[13, 19, 20, 22],
        "test":[17, 21, 23, 24]
}
SEED_IV_session3_dataset_partition = {
        "train":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 19, 20],
        "val":[11, 15, 17, 22],
        "test":[18, 21, 23, 24]
}


SEED_IV_class_emotions = ["neutral", "sad", "fear", "happy"]
SEED_IV_class_labels = [0, 1, 2, 3]

DEAP_class_emotions = ["LV", "HV"]
DEAP_class_labels = [0, 1]
