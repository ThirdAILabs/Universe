from thirdai.distributed_bolt import DistributedBolt

# config_filename = './amzn670k_distributed.txt'
config_filename = "./mnist.txt"
head = DistributedBolt(2, config_filename)
head.train(
    circular=False, compression="DRAGON", compression_density=0.10, scheduler=True
)
acc, _ = head.predict()
print(acc["categorical_accuracy"])
