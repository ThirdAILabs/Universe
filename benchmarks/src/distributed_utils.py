import os

import tqdm


def split_into_2(
    file_to_split, destination_file_1, destination_file_2, with_header=False
):
    with open(file_to_split, "r") as input_file:
        with open(destination_file_1, "w+") as f_1:
            with open(destination_file_2, "w+") as f_2:
                for i, line in enumerate(input_file):
                    if with_header and i == 0:
                        f_1.write(line)
                        f_2.write(line)
                        continue
                    if i % 2 == 0:
                        f_1.write(line)
                    else:
                        f_2.write(line)


def setup_ray(num_workers=2):
    import ray
    import thirdai.distributed_bolt as dist

    # reserve one CPU for Ray Trainer
    num_cpu_per_node = (dist.get_num_cpus() - 1) // num_workers

    assert num_cpu_per_node > 0, "Number of CPUs per node should be greater than 0"
    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "env_vars": {
                "OMP_NUM_THREADS": f"{num_cpu_per_node}",
                "PYTHONPATH": os.path.join(
                    working_dir, "../../"
                ),  # Change to your home directory where benchmarks module is present
            },
        },
        ignore_reinit_error=True,
    )
    scaling_config = ray.train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=False,
        resources_per_worker={"CPU": num_cpu_per_node},
        placement_strategy="PACK",
    )
    return scaling_config


def create_udt_model(n_classes, output_dim, num_hashes, embedding_dimension):
    from thirdai import bolt

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(
                delimiter=":", n_classes=n_classes, type="int"
            ),
        },
        target="DOC_ID",
        embedding_dimension=embedding_dimension,
        extreme_output_dim=output_dim,
        extreme_num_hashes=num_hashes,
    )
    return model


def test_ndb(db, df):
    scores = {"top_1": 0, "top_3": 0, "top_5": 0, "top_10": 0}
    total_count = len(df)

    for actual_id, row in tqdm.tqdm(df.iterrows(), desc="Progress", total=total_count):
        query = row["TITLE"]

        search_results = db.search(query=query, top_k=10)
        all_retrieved_ids = [int(result.id) for result in search_results]

        for k in [1, 3, 5, 10]:
            if actual_id in all_retrieved_ids[:k]:
                scores[f"top_{k}"] += 1

    score = {f"top_{k}": scores[f"top_{k}"] / total_count for k in [1, 3, 5, 10]}
    return score
