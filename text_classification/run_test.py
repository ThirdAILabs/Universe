
from cookie_monster import *


def test_cookie(input_dim, param_file_first_hidden, param_file_second_hidden, test_dir):
    model = CookieMonster(
        input_dim, hidden_dimension=2048, hidden_sparsity=0.1, mlflow_enabled=True
    )
    #model.load_hidden_layer_parameters(param_file_first_hidden, param_file_second_hidden)
    model.evaluate(test_dir, evaluate=False)


def test_scratch(train_file, test_file, label_dim, epochs, lr, mlflow_uri):
    input_layer = bolt.graph.Input(dim=100000)

    hidden_layer = bolt.graph.FullyConnected(
        dim=2048,
        sparsity=0.1,
        activation="relu",
    )(input_layer)

    second_hidden_layer = bolt.graph.FullyConnected(
        dim=1024,
        sparsity=0.1,
        activation="relu",
    )(hidden_layer)

    output_layer = bolt.graph.FullyConnected(dim=label_dim, activation="softmax")(
        second_hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    train_x, train_y = load_scratch_data(train_file, label_dim)
    test_x, test_y = load_scratch_data(test_file, label_dim)

    train_config = bolt.graph.TrainConfig.make(learning_rate=lr, epochs=1)

    predict_config = bolt.graph.PredictConfig.make().with_metrics(
        ["categorical_accuracy", "mean_squared_error"]
    )

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Cookie Monster - Results Validation")

    first_epoch = False
    for epoch in range(epochs):
        model.train(train_x, train_y, train_config)

        if epoch == 1:
            # Save model parameters to MlFlow
            first_hidden_layer_save_loc = "./first_hidden_layer_parameters"
            second_hidden_layer_save_loc = "./second_hidden_layer_parameters"
            hidden_layer.save_parameters(first_hidden_layer_save_loc)
            second_hidden_layer.save_parameters(second_hidden_layer_save_loc)
            mlflow.log_artifact(first_hidden_layer_save_loc)
            mlflow.log_artifact(second_hidden_layer_save_loc)

        model.predict(test_x, test_y, predict_config)


def load_scratch_data(file, label_dim):
    pipeline = DataPipeline(
        file,
        batch_size=64,
        input_blocks=[blocks.Text(1, text_encodings.PairGram(100000))],
        label_blocks=[blocks.Categorical(0, label_dim)],
        delimiter=",",
    )
    data, label = pipeline.load_in_memory()
    return data, label


if __name__ == "__main__":
    uri = (
        "http://deplo-mlflo-15qe25sw8psjr-1d20dd0c302edb1f.elb.us-east-1.amazonaws.com"
    )

    # mlflow.start_run(run_name="train_run")
    # test_scratch(
    #     train_file="/share/henry/bert_tokenized_csvs/yelp_polarity_train.csv",
    #     test_file="/share/henry/bert_tokenized_csvs/yelp_polarity_test.csv",
    #     label_dim=2,
    #     epochs=10,
    #     lr=0.0001,
    #     mlflow_uri=uri
    # )

    # test_cookie(
    #     100000,
    #     "/home/blaise/cookie-monster/CookieMonster_pretrained_2k_0.1/wiki_20M",
    #     "/home/blaise/cookie-monster/yelp_polarity_config/",
    # )

    test_cookie(
        100000,
        "s3://mlflow-artifacts-199696198976/29/96096d3e83bc468aad2926c3cb4f6620/artifacts/first_hidden_layer_parameters",
        "s3://mlflow-artifacts-199696198976/29/96096d3e83bc468aad2926c3cb4f6620/artifacts/second_hidden_layer_parameters",
        "/home/blaise/cookie-monster/wikipedia_config"
    )
    # test_cookie(100000, "/home/henry/cookie_models/amz_yelp_tomatoes_dbpedia_sst_ag", "/home/henry/experiments/cookie_basic/test/")

    # test_scratch(
    #     train_file="sentences_tokenized_shuffled_trimmed_40M.txt",
    #     test_file="/share/data/BERT/sentences_tokenized_shuffled_trimmed_test_100k.txt",
    #     label_dim=30224,
    #     epochs=1,
    #     lr=0.0001
    # )
