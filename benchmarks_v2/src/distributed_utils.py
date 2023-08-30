import os


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


def setup_ray():
    import ray
    import thirdai.distributed_bolt as dist

    # reserve one CPU for Ray Trainer
    num_cpu_per_node = (dist.get_num_cpus() - 1) // 2

    assert num_cpu_per_node > 0, "Number of CPUs per node should be greater than 0"
    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "env_vars": {
                "OMP_NUM_THREADS": f"{num_cpu_per_node}",
                "PYTHONPATH": os.path.join(
                    working_dir, "../../"
                ),  # Change to your home directory where benchmarks_v2 module is present
            },
        },
        ignore_reinit_error=True,
    )
    scaling_config = ray.air.ScalingConfig(
        num_workers=2,
        use_gpu=False,
        resources_per_worker={"CPU": num_cpu_per_node},
        placement_strategy="PACK",
    )
    return scaling_config


def create_udt_model(n_target_classes, output_dim, num_hashes, embedding_dimension):
    from thirdai import bolt

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
        options={
            "embedding_dimension": embedding_dimension,
            "extreme_output_dim": output_dim,
            "extreme_num_hashes": num_hashes,
        },
    )
    return model
