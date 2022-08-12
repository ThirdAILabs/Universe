
from train_distributed_amzn670k import train_model
import ray

def main():
    logfile = "benchmarking"
    compression_schemes = ["topk", "DRAGON", "UNBIASED_DRAGON"]
    compression_density = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    scheduler = False
    models = ["amazon_polarity", "yelp"]
    # models=["mnist"]
    for model in models:
        print(f"inside benchmarking script with model {model}")
        train_model(
            model=model,
            compression_scheme="None",
            logfile=f"benchmarking_{model}.log",
            scheduler=False,
            compression_density=1,
        )

        for schemes in compression_schemes:
            for density in compression_density:
                train_model(
                    model=model,
                    compression_scheme=schemes,
                    compression_density=density,
                    logfile=f"benchmarking_{model}_{schemes}_{density}.log",
                    scheduler=False,
                )
            

if __name__ == "__main__":
    main()
