from thirdai.distributed_bolt import DistributedBolt

import argparse
import ray

def train_model(
    model, compression_scheme, logfile, scheduler=False, compression_density=1
):
    print(f"Running the model {model}")
    if model == "yelp":
        config_filename = "./yelp_polarity.txt"
    elif model == "amazon":
        config_filename = "./amzn670k_distributed.txt"
    elif model == "mnist":
        config_filename = "./mnist.txt"
    elif model == "amazon_polarity":
        config_filename = "./amazon_polarity.txt"
    else:
        raise Exception("Invalid model name provided")

    print("Training the following model " + config_filename.split(".")[1][1:])

    if logfile is None:
        logfile = f"logfile_experiments_{config_filename.split('.')[1][1:]}.log"

    print("The logfile is:"+logfile)
    head = DistributedBolt(
        2,
        config_filename,
        pregenerate=True,
        logfile=logfile,
    )

    if compression_scheme == "None":
        head.train(
            circular=False,
            compression=None,
        )
    else:
        head.train(
            circular=False,
            compression=compression_scheme,
            compression_density=compression_density,
            scheduler=scheduler,
        )
    ray.shutdown()
        


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        default="amazon_polarity",
        type=str,
        help="The model file should be in the same directory as this file",
    )
    parser.add_argument(
        "-c",
        "--compression_scheme",
        default="None",
        type=str,
        help="Specify the compression scheme",
    )
    parser.add_argument(
        "-d",
        "--compression_density",
        default=0.1,
        type=float,
        help="Specify the compression density. Not properly tested. Can give segfaults at very low density",
    )
    parser.add_argument(
        "-l", "--logfile_name", default=None, type=str, help="logfile name"
    )
    parser.add_argument(
        "-s", "--scheduler", default=False, type=bool, help="Make this false generally"
    )

    args = vars(parser.parse_args())

    model = args["model"]
    compression_scheme = args["compression_scheme"]
    compression_density = args["compression_density"]
    logfile = args["logfile_name"]
    scheduler = args["scheduler"]

    train_model(
        model=model,
        compression_scheme=compression_scheme,
        compression_density=compression_density,
        logfile=logfile,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
