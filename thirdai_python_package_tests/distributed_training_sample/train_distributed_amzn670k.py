from thirdai.distributed_bolt import DistributedBolt

import sys

if sys.argv[1] == "yelp":
    config_filename = "./yelp_polarity.txt"
elif sys.argv[1] == "amazon":
    config_filename = "./amzn670k_distributed.txt"
elif sys.argv[1] == "mnist":
    config_filename = "./mnist.txt"
else:
    config_filename = "./amazon_polarity.txt"

print(config_filename.split(".")[1][1:])

# pregenerating random numbers for UNBIASED_DRAGON

head = DistributedBolt(
    2,
    config_filename,
    pregenerate=True,
    logfile=f"logfile_experiments_{config_filename.split('.')[1][1:]}.log",
)

if sys.argv[2] == "None":
    head.train(
        circular=False,
        compression=None,
    )
else:
    head.train(
        circular=False,
        compression=sys.argv[2],
        compression_density=0.05,
        scheduler=False,
    )
acc, _ = head.predict()
print(acc["categorical_accuracy"])
