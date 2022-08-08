from thirdai.distributed_bolt import DistributedBolt

<<<<<<< HEAD
config_filename = './amzn670k_distributed.txt'
head = DistributedBolt(8, config_filename) 
head.train(circular=True)
acc , _ = head.predict()
print(acc["categorical_accuracy"])
=======
config_filename = "./amzn670k_distributed.txt"
head = DistributedBolt(2, config_filename, num_cpus_per_node=20)
head.train(circular=False)
acc, _ = head.predict()
print(acc["categorical_accuracy"])
>>>>>>> 43210961db5709b41bd2f44cefb45e4569313de6
