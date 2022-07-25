from thirdai.distributed_bolt import DistributedBolt

config_filename = './amzn670k_distributed.txt'
head = DistributedBolt(1, config_filename) 
head.train(circular=False,compression="DRAGON",compression_density=0.99)
acc , _ = head.predict()
print(acc["categorical_accuracy"])