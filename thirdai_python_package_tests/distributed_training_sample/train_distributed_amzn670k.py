from thirdai.distributed_bolt import DistributedBolt

config_filename = "./amzn670k_distributed.txt"
head = DistributedBolt(2, config_filename, num_cpus_per_node=20)
head.train(circular=False)
acc, _ = head.predict()
print(acc["categorical_accuracy"])
