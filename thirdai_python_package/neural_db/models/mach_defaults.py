# This file contains default functions and variables that Mach uses.

metric_to_track = "hash_precision@5"
acc_to_stop = 0.95


def autotune_from_scratch_min_max_epochs(size):
    if size < 10000:
        return 10, 15
    if size < 100000:
        return 5, 10
    if size < 1000000:
        return 3, 8
    return 1, 5


def autotune_from_base_min_max_epochs(size):
    if size < 100000:
        return 5, 10
    if size < 1000000:
        return 3, 8
    return 1, 5


def training_arguments_from_scratch(size):
    min_epochs, max_epochs = autotune_from_scratch_min_max_epochs(size)

    # 0.005 was an artifact of the original pocketllm/playground which was
    # tested with small docs but not yet thoroughly benchmarked
    # 0.001 is what we use for all of our benchmarks and gives us the best numbers
    learning_rate = 0.005 if size < 1000 else 0.001
    freeze_before_train = False

    return {
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "freeze_before_train": freeze_before_train,
    }


def training_arguments_from_base(size):
    min_epochs, max_epochs = autotune_from_base_min_max_epochs(size)
    learning_rate = 0.001
    freeze_before_train = True

    return {
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "freeze_before_train": freeze_before_train,
    }
