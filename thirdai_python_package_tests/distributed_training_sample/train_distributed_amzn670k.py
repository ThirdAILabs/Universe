from thirdai.distributed_bolt import DistributedBolt

import sys
config_filename = "./amzn670k_distributed.txt"
# config_filename = "./mnist.txt"
head = DistributedBolt(2, config_filename, pregenerate=False)
if sys.argv[1]=="None":
    head.train(
        circular=False, compression=None, compression_density=0.10, scheduler=True
    )
else:
    head.train(
        circular=False, compression=sys.argv[1], compression_density=0.10, scheduler=False
    )
acc, _ = head.predict()
print(acc["categorical_accuracy"])
