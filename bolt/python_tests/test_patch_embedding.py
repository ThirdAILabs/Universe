import numpy as np
import pytest
from thirdai import bolt


def build_model(patches, patch_emb):
    loss = bolt.nn.losses.BinaryCrossEntropy(
        patch_emb, labels=bolt.nn.Input(patch_emb.dim())
    )

    return bolt.nn.Model(inputs=[patches], outputs=[], losses=[loss])


BATCH_SIZE = 4
N_PATCHES = 8
PATCH_DIM = 6
EMB_DIM = 20


@pytest.mark.unit
@pytest.mark.parametrize("emb_nonzeros", [EMB_DIM, EMB_DIM // 4])
def test_patch_embedding(emb_nonzeros):
    concat_patches = bolt.nn.Input(PATCH_DIM * N_PATCHES)
    concat_patch_emb_op = bolt.nn.PatchEmbedding(
        emb_dim=EMB_DIM,
        patch_dim=PATCH_DIM,
        n_patches=N_PATCHES,
        sparsity=emb_nonzeros / EMB_DIM,
        activation="linear",
    )
    concat_patch_emb = concat_patch_emb_op(concat_patches)

    concat_model = build_model(concat_patches, concat_patch_emb)

    single_patches = bolt.nn.Input(PATCH_DIM)
    single_patch_emb_op = bolt.nn.FullyConnected(
        dim=EMB_DIM,
        input_dim=PATCH_DIM,
        sparsity=1.0,
        activation="linear",
    )
    single_patch_emb = single_patch_emb_op(single_patches)

    single_model = build_model(single_patches, single_patch_emb)

    concat_patch_emb_op.set_weights(single_patch_emb_op.weights)
    concat_patch_emb_op.set_biases(single_patch_emb_op.biases)

    patches = np.random.rand(BATCH_SIZE, N_PATCHES, PATCH_DIM)

    labels = np.random.rand(BATCH_SIZE, N_PATCHES, EMB_DIM)

    concat_patch_tensor = bolt.nn.Tensor(
        patches.reshape(BATCH_SIZE, -1), with_grad=True
    )
    concat_patch_labels = bolt.nn.Tensor(labels.reshape(BATCH_SIZE, -1))

    single_patch_tensor = bolt.nn.Tensor(patches.reshape(-1, PATCH_DIM), with_grad=True)
    single_patch_labels = bolt.nn.Tensor(labels.reshape(-1, EMB_DIM))

    concat_model.train_on_batch([concat_patch_tensor], [concat_patch_labels])

    single_model.train_on_batch([single_patch_tensor], [single_patch_labels])

    if emb_nonzeros < EMB_DIM:
        emb_indices = concat_patch_emb.tensor().active_neurons.reshape(-1, emb_nonzeros)
        emb_values = concat_patch_emb.tensor().activations.reshape(-1, emb_nonzeros)

        expected_embeddings = single_patch_emb.tensor().activations

        assert len(emb_indices) == len(expected_embeddings)
        assert len(emb_values) == len(expected_embeddings)

        for i in range(len(emb_indices)):
            assert np.allclose(emb_values[i], expected_embeddings[i][emb_indices[i]])

    else:
        assert np.array_equal(
            concat_patch_emb.tensor().activations,
            single_patch_emb.tensor().activations.reshape(BATCH_SIZE, -1),
        )

        # Multiply by N_PATCHES since loss function scales by batch size.
        assert np.array_equal(
            concat_patches.tensor().gradients,
            N_PATCHES * single_patches.tensor().gradients.reshape(BATCH_SIZE, -1),
        )
