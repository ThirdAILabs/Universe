import numpy as np
import pytest
import thirdai
from test_bolt_v2_save_load import (
    N_CLASSES,
    evaluate_model,
    gen_numpy_training_data,
    get_data,
    get_model,
)
from thirdai import bolt_v2 as bolt
from thirdai import dataset


def average_values(trainers, get_func, set_func):
    values = [np.array(get_func(trainer.model)) for trainer in trainers]
    avg_values = sum(values) / len(values)
    for trainer in trainers:
        set_func(trainer.model, avg_values)


def equal_model_paramters(trainers):
    params = [np.array(trainer.model.get_parameters()) for trainer in trainers]
    return np.allclose(params[0], params[1])


# When training in distributed, sparse parameters updates might not update all
# the gradients updated by all-reduce
def disable_sparse_updates(trainers):
    for trainer in trainers:
        trainer.model.disable_sparse_parameter_updates()


# We dont have pygloo wheels working in release, So, we can't have a integration tests.
# TODO(pratik): remove this test, once we have a integration test using pygloo,or we
# have implemented internal support for gloo.
@pytest.mark.unit
def test_multiple_trainers():
    EPOCHS = 1

    import copy

    model_1 = get_model()
    model_2 = copy.deepcopy(model_1)

    train_data_1, train_labels_1, test_data, test_labels_np = get_data()

    trainers = [bolt.train.Trainer(model_1), bolt.train.Trainer(model_2)]

    disable_sparse_updates(trainers)

    # Training them on same data should still get different
    # gradients as we are training with sparsity
    for _ in range(EPOCHS):
        for x, y in zip(train_data_1, train_labels_1):
            for trainer in trainers:
                trainer.model.train_on_batch(x, y)

            # averages model gradients
            average_values(
                trainers,
                lambda model: model.get_gradients(),
                lambda model, values: model.set_gradients(values),
            )

            for trainer in trainers:
                trainer.model.update_parameters(learning_rate=0.05)

    # assert equal_model_paramters(trainers), "Trainer models are not the same."
    evaluate_model(trainers[0].model, test_data, test_labels_np)
    evaluate_model(trainers[1].model, test_data, test_labels_np)
