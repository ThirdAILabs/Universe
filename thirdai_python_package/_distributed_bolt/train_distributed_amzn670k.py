from thirdai.distributed_bolt import DistributedBolt

config_filename = './amzn670k_distributed.txt'
head = DistributedBolt(['3','11'], config_filename) 
head.train(False)
print(head.predict())