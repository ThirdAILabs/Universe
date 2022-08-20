import thirdai.distributed_bolt as db


if __name__ == "__main__":
    config_filename = "./default_config.txt"
    head = db.FullyConnectedNetwork(
        no_of_workers=4,
        config_filename=config_filename,
        num_cpus_per_node=48,
    )
    head.train(circular= True)
    acc, _ = head.predict()
    print(acc["categorical_accuracy"])
