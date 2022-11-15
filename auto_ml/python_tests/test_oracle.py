import pytest
from thirdai import bolt, deployment

pytestmark = [pytest.mark.unit]

CONFIG_FILE = "./saved_udt_config"
TRAIN_FILE = "tempTrainFile.csv"
TEST_FILE = "tempTestFile.csv"


def make_serialized_udt_config():
    model_config = deployment.ModelConfig(
        input_names=["input"],
        nodes=[
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.ConstantParameter(512),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input",
            ),
            deployment.FullyConnectedNodeConfig(
                name="output",
                dim=deployment.DatasetLabelDimensionParameter(),
                sparsity=deployment.AutotunedSparsityParameter(
                    deployment.DatasetLabelDimensionParameter.dimension_param_name
                ),
                activation=deployment.ConstantParameter("softmax"),
                predecessor="hidden",
            ),
        ],
        loss=bolt.nn.losses.CategoricalCrossEntropy(),
    )

    dataset_config = deployment.UDTDatasetFactory(
        config=deployment.UserSpecifiedParameter("config", type=bolt.models.UDTConfig),
        force_parallel=deployment.ConstantParameter(False),
        text_pairgram_word_limit=deployment.ConstantParameter(15),
        contextual_columns=deployment.ConstantParameter(False),
    )

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=2048,
        freeze_hash_tables=True,
    )

    config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

    config.save(CONFIG_FILE)


def write_lines_to_file(file, lines):
    with open(file, "w") as f:
        for line in lines:
            f.writelines(line + "\n")


def make_simple_udt_model():
    make_serialized_udt_config()

    write_lines_to_file(
        TRAIN_FILE,
        [
            "userId,movieId,timestamp",
            "0,0,2022-08-29",
            "1,0,2022-08-30",
            "1,1,2022-08-31",
            "1,2,2022-09-01",
        ],
    )

    write_lines_to_file(
        TEST_FILE,
        [
            "userId,movieId,timestamp",
            "0,1,2022-08-31",
            "2,0,2022-08-30",
        ],
    )

    model = bolt.models.Pipeline(
        config_path=CONFIG_FILE,
        parameters={
            "config": bolt.models.UDTConfig(
                data_types={
                    "userId": bolt.types.categorical(),
                    "movieId": bolt.types.categorical(),
                    "timestamp": bolt.types.date(),
                },
                temporal_tracking_relationships={"userId": ["movieId"]},
                target="movieId",
                n_target_classes=3,
            )
        },
    )

    return model


def test_udt_save_load():
    model = make_simple_udt_model()

    train_config = bolt.TrainConfig(epochs=2, learning_rate=0.01)
    model.train(TRAIN_FILE, train_config, batch_size=2048)
    model.save("saveLoc")
    before_load_output = model.evaluate(TEST_FILE)
    model = bolt.models.Pipeline.load("saveLoc")
    after_load_output = model.evaluate(TEST_FILE)

    assert (before_load_output == after_load_output).all()


def test_multiple_predict_returns_same():
    model = make_simple_udt_model()
    train_config = bolt.TrainConfig(epochs=2, learning_rate=0.01)
    model.train(TRAIN_FILE, train_config, batch_size=2048)

    sample = "0,,2022-08-31"
    prev_result = model.predict(sample)
    for _ in range(5):
        result = model.predict(sample)
        assert (prev_result == result).all()
        prev_result = result


def test_explanations_total_percentage():
    model = make_simple_udt_model()
    train_config = bolt.TrainConfig(epochs=2, learning_rate=0.01)
    model.train(TRAIN_FILE, train_config, batch_size=2048)

    sample = "0,,2022-08-31"
    explanations = model.explain(sample)
    total_percentage = 0
    for explanation in explanations:
        total_percentage += abs(explanation.percentage_significance)

    assert total_percentage > 99.99


def test_index_changes_predict():
    model = make_simple_udt_model()
    context = model.get_data_processor()
    train_config = bolt.TrainConfig(epochs=2, learning_rate=0.01)
    model.train(TRAIN_FILE, train_config, batch_size=2048)

    sample = "0,,2022-08-31"

    first_result = model.predict(sample)

    context.update_temporal_trackers("0,1,2022-08-31")

    second_result = model.predict(sample)

    assert (first_result != second_result).any()


def test_context_serialization():
    model = make_simple_udt_model()
    context = model.get_data_processor()
    train_config = bolt.TrainConfig(epochs=2, learning_rate=0.01)
    model.train(TRAIN_FILE, train_config, batch_size=2048)

    model.save("saveLoc")
    saved_model = bolt.models.Pipeline.load("saveLoc")
    saved_context = saved_model.get_data_processor()

    sample = "0,,2022-08-31"
    update = "0,1,2022-08-31"

    context.update_temporal_trackers(update)
    saved_context.update_temporal_trackers(update)

    original_model_result_after_context_update = model.predict(sample)
    saved_model_result_after_context_update = saved_model.predict(sample)

    assert (
        original_model_result_after_context_update
        == saved_model_result_after_context_update
    ).any()
