# import os
# import shutil

# import numpy as np
# import pytest
# from distributed_utils import setup_ray
# from ray.train import RunConfig
# from seismic_dataset_fixtures import classification_dataset, subcube_dataset
# from thirdai import bolt


# def train_distributed_seismic(model, data_path, subcube_shape, emb_dim):
#     scaling_config = setup_ray()

#     log_file = "seismic_log"
#     checkpoint_dir = "seismic_checkpoints"
#     model.train_distributed(
#         data_path=data_path,
#         learning_rate=0.0001,
#         epochs=2,
#         batch_size=8,
#         scaling_config=scaling_config,
#         run_config=RunConfig(storage_path="~/ray_results"),
#         log_file=log_file,
#         checkpoint_dir=checkpoint_dir,
#     )

#     n_cubes_to_embed = 3
#     subcubes_to_embed = np.random.rand(n_cubes_to_embed, *subcube_shape).astype(
#         np.float32
#     )

#     embeddings = model.embeddings(subcubes_to_embed)

#     assert embeddings.shape == (n_cubes_to_embed, emb_dim)

#     assert len(os.listdir(checkpoint_dir)) == 2
#     assert os.path.exists(log_file)
#     assert os.path.exists(log_file + ".worker_1")

#     shutil.rmtree(checkpoint_dir)
#     os.remove(log_file)
#     os.remove(log_file + ".worker_1")


# @pytest.mark.distributed
# def test_distributed_seismic_embedding_model(subcube_dataset):
#     subcube_directory, subcube_shape, patch_shape = subcube_dataset
#     emb_dim = 256
#     model = bolt.seismic.SeismicEmbedding(
#         subcube_shape=subcube_shape[0],
#         patch_shape=patch_shape[0],
#         embedding_dim=emb_dim,
#         size="small",
#         max_pool=2,
#     )

#     train_distributed_seismic(
#         model=model,
#         data_path=subcube_directory,
#         subcube_shape=subcube_shape,
#         emb_dim=emb_dim,
#     )


# @pytest.mark.distributed
# def test_distributed_seismic_classification(classification_dataset):
#     sample_index_file, subcube_shape, patch_shape, n_classes = classification_dataset
#     emb_dim = 256
#     model = bolt.seismic.SeismicEmbedding(
#         subcube_shape=subcube_shape[0],
#         patch_shape=patch_shape[0],
#         embedding_dim=emb_dim,
#         size="small",
#         max_pool=2,
#     )

#     model = bolt.seismic.SeismicClassifier(model, n_classes=n_classes)

#     train_distributed_seismic(
#         model=model,
#         data_path=sample_index_file,
#         subcube_shape=subcube_shape,
#         emb_dim=emb_dim,
#     )
