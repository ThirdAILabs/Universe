from thirdai.distributed_bolt import DistributedBolt

import sys

if sys.argv[2]=="yelp":
    config_filename="./yelp_polarity.txt"
elif sys.argv[2]=="amazon":
    config_filename= "./amzn670k_distributed.txt"
elif sys.argv[2]=="mnist":
    config_filename = "./mnist.txt"
else:
    config_filename="./amazon_polarity.txt"

head = DistributedBolt(2, config_filename, pregenerate=True, logfile=f"logfile_shubh_{sys.argv[2]}.log")

if sys.argv[1]=="None":
    head.train(
        circular=False, compression=None, compression_density=0.10, scheduler=False
    )
else:
    head.train(
        circular=False, compression=sys.argv[1], compression_density=0.10, scheduler=False
    )
acc, _ = head.predict()
print(acc["categorical_accuracy"])
