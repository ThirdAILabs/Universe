from thirdai.distributed_bolt import DistributedBolt

config_filename = './amzn670k_distributed.txt'
head = DistributedBolt(2, config_filename) 
head.train(circular=True)
acc , _ = head.predict()
print(acc["categorical_accuracy"])