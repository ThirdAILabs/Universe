import thirdai.distributed_bolt as DistributedBolt


if __name__ == "__main__":
    config_filename = "./default_config.txt"
    head = DistributedBolt.FullyConnectedNetwork(
        no_of_workers=2,
        config_filename=config_filename,
        num_cpus_per_node=20,
    )
    head.train(circular=False)
    acc, _ = head.predict()
    print(acc["categorical_accuracy"])
