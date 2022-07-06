from cookie_monster import *

if __name__ == "__main__":
    layers = [
        bolt.FullyConnected(
            dim=2000, sparsity=0.15, activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=4,
                num_tables=50,
                range_pow=4 * 3,
                reservoir_size=32,
            )
        ),
        bolt.FullyConnected(dim=2, activation_function=bolt.ActivationFunctions.Softmax),
    ]

    cookie_model = CookieMonster(layers)
    cookie_model.train_corpus("/home/henry/cookie_train/", mlflow_enabled=True, evaluate=False)
    cookie_model.evaluate("/home/henry/cookie_test/")

    # Evaluate
    # cookie_model.download_hidden_weights("s3://mlflow-artifacts-199696198976/29/0329c743e3374013b626fdb931a678f9/artifacts/weights.npy", 
    #     "s3://mlflow-artifacts-199696198976/29/0329c743e3374013b626fdb931a678f9/artifacts/biases.npy")
