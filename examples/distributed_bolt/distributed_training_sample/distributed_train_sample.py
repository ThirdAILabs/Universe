import thirdai.distributed_bolt as db


if __name__ == "__main__":
    config_filename = "./default_config.txt"
    head = db.FullyConnectedNetwork(
        num_workers=2,
        config_filename=config_filename,
        num_cpus_per_node=20,
        communication_type="circular",
    )
    head.train()
    metrics = head.predict()
    print(metrics[0]["categorical_accuracy"])
