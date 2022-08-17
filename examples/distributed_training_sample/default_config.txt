from thirdai.distributed_bolt import DistributedBolt


if __name__ == "__main__":
    config_filename = "./default_config.txt"
    head = DistributedBolt(
        no_of_workers=no_of_worker, config_filename=config_filename, num_cpus_per_node=cpus_for_each_worker
    )
    head.train(circular=True)
    acc, _ = head.predict()
    print(acc["categorical_accuracy"])
