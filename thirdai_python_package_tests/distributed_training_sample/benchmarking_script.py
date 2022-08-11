from train_distributed_amzn670k import train_model


def main():

    logfile = "benchmarking"
    compression_schemes = ["topk", "DRAGON", "UNBIASED_DRAGON"]
    compression_density = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    scheduler = False
    models = ["amazon_polarity", "yelp_polarity"]

    for model in models:
        train_model(
            model=model,
            compression_scheme="None",
            logfile=f"benchmarking_{models}.log",
            scheduler=False,
            compression_density=1,
        )
        for schemes in compression_schemes:
            for density in compression_density:
                train_model(
                    model=model,
                    compression_scheme=schemes,
                    compression_density=density,
                    logfile=f"benchmarking_{models}.log",
                    scheduler=False,
                )


if __name__ == main():
    main()
