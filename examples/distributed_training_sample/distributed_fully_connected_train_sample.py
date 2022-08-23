import thirdai.distributed_bolt as db


if __name__ == "__main__":
    config_filename = "./default_config.txt"
    head = db.FullyConnectedNetwork(
        no_of_workers=4,
        config_filename=config_filename,
        num_cpus_per_node=48,
    )
<<<<<<< HEAD:examples/distributed_training_sample/distributed_fully_connected_train_sample.py
    head.train(circular= True)
=======
    head.train(circular=True)
>>>>>>> db42f91c45af737f69ad309e612136fe0e3c0070:examples/distributed_training_sample/distributed_train_sample.py
    acc, _ = head.predict()
    print(acc["categorical_accuracy"])
