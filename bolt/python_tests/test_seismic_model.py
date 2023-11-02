import os

import numpy as np
import pandas as pd
import pytest
import torch
from seismic_dataset_fixtures import classification_dataset, subcube_dataset
from thirdai import bolt


@pytest.mark.unit
@pytest.mark.parametrize("max_pool", [None, 2])
def test_seismic_embedding_model(subcube_dataset, max_pool):
    subcube_directory, subcube_shape, patch_shape = subcube_dataset
    emb_dim = 100
    model = bolt.seismic.SeismicEmbedding(
        subcube_shape=subcube_shape[0],
        patch_shape=patch_shape[0],
        embedding_dim=emb_dim,
        size="small",
        max_pool=max_pool,
    )

    class Validation:
        def __init__(self):
            self.invocation_cnt = 0

        def __call__(self, model):
            self.invocation_cnt += 1

    validation_fn = Validation()

    model.train(
        subcube_directory=subcube_directory,
        learning_rate=0.0001,
        epochs=1,
        batch_size=8,
        validation_fn=validation_fn,
    )

    assert validation_fn.invocation_cnt == 1

    n_cubes_to_embed = 4
    subcubes_to_embed = (
        np.random.rand(n_cubes_to_embed, *subcube_shape).astype(np.float32) / 10
    )

    embs = model.embeddings(subcubes_to_embed, sparse_inference=True)

    assert embs.shape == (n_cubes_to_embed, emb_dim)

    eval_dir = "./eval_subcubes"
    os.makedirs(eval_dir, exist_ok=True)

    np.save(os.path.join(eval_dir, "tgt.npy"), subcubes_to_embed[0])
    for i in range(n_cubes_to_embed):
        np.save(os.path.join(eval_dir, f"candidate_{i}.npy"), subcubes_to_embed[i])

    sims = model.score_subcubes(eval_dir)

    assert set([x[0] for x in sims]) == set(
        f"candidate_{i}.npy" for i in range(n_cubes_to_embed)
    )


@pytest.mark.unit
def test_seismic_embedding_finetuning(subcube_dataset):
    subcube_directory, subcube_shape, patch_shape = subcube_dataset

    emb_dim = 100

    model = bolt.seismic.SeismicEmbedding(
        subcube_shape=subcube_shape[0],
        patch_shape=patch_shape[0],
        embedding_dim=emb_dim,
        size="small",
        max_pool=2,
    )

    embs = model.forward(torch.rand(5, *subcube_shape))
    assert embs.shape == (5, emb_dim)
    assert embs.requires_grad
    model.backpropagate(torch.rand(*embs.shape))
    model.update_parameters(0.001)

    embs = model.embeddings(np.random.rand(3, *subcube_shape))
    assert embs.shape == (3, emb_dim)

    with pytest.raises(
        ValueError,
        match="Can not use unsupervised pretraining on a model after using "
        "finetuning since the decoder has been invalidated.",
    ):
        model.train(
            subcube_directory=subcube_directory,
            learning_rate=0.0001,
            epochs=1,
            batch_size=8,
        )


@pytest.mark.unit
def test_seismic_classifier(classification_dataset):
    sample_index, subcube_shape, patch_shape, n_classes = classification_dataset

    emb_dim = 100
    emb_model = bolt.seismic.SeismicEmbedding(
        subcube_shape=subcube_shape[0],
        patch_shape=patch_shape[0],
        embedding_dim=emb_dim,
        size="small",
        max_pool=2,
    )

    classifier = bolt.seismic.SeismicClassifier(emb_model, n_classes=n_classes)

    classifier.train(
        sample_index_file=sample_index,
        learning_rate=0.0001,
        epochs=1,
        batch_size=8,
    )

    n_cubes_to_embed = 2
    subcubes_to_embed = (
        np.random.rand(n_cubes_to_embed, *subcube_shape).astype(np.float32) / 10
    )

    assert np.array_equal(
        emb_model.embeddings(subcubes_to_embed),
        classifier.embeddings(subcubes_to_embed),
    )

    predictions = classifier.predict(subcubes_to_embed, sparse_inference=True)
    assert predictions.shape == (n_cubes_to_embed, n_classes)


@pytest.mark.unit
def test_create_patches():
    integer_cube = torch.arange(512).reshape((8, 8, 8)).type(torch.float32)

    cubes = torch.stack([integer_cube, 512 + integer_cube, 1024 + integer_cube])

    patches = bolt.seismic_modifications.convert_to_patches(
        cubes, expected_subcube_shape=(8, 8, 8), patch_shape=(4, 4, 4)
    )

    patch = 0
    ranges = [(0, 4), (4, 8)]
    for x1, x2 in ranges:
        for y1, y2 in ranges:
            for z1, z2 in ranges:
                assert np.array_equal(
                    integer_cube[x1:x2, y1:y2, z1:z2].flatten(), patches[0][patch]
                )
                patch += 1

    assert np.array_equal(512 + patches[0], patches[1])
    assert np.array_equal(1024 + patches[0], patches[2])


@pytest.mark.unit
def test_create_patches_max_pool():
    integer_cube = torch.arange(512).reshape((8, 8, 8)).type(torch.float32)

    cubes = torch.stack([integer_cube, 512 + integer_cube, 1024 + integer_cube])

    patches = bolt.seismic_modifications.convert_to_patches(
        cubes,
        expected_subcube_shape=(8, 8, 8),
        patch_shape=(4, 4, 4),
        max_pool=(2, 2, 2),
    )

    pooled_cube = np.zeros((4, 4, 4))
    for x in range(4):
        for y in range(4):
            for z in range(4):
                pooled_cube[x, y, z] = integer_cube[2 * x + 1, 2 * y + 1, 2 * z + 1]

    patch = 0
    ranges = [(0, 2), (2, 4)]
    for x1, x2 in ranges:
        for y1, y2 in ranges:
            for z1, z2 in ranges:
                assert np.array_equal(
                    pooled_cube[x1:x2, y1:y2, z1:z2].flatten(), patches[0][patch]
                )
                patch += 1

    assert np.array_equal(512 + patches[0], patches[1])
    assert np.array_equal(1024 + patches[0], patches[2])
