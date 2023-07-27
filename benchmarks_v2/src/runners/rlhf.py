import mlflow
import numpy as np

from ..configs.rlhf_configs import RlhfConfig
from .runner import Runner


class RlhfRunner(Runner):
    config_type = RlhfConfig

    @classmethod
    def run_benchmark(cls, config: RlhfConfig, path_prefix: str, mlflow_logger):
        # 1. Load/preprocess data and pretrain the model.
        config.prepare_data(path_prefix=path_prefix)
        model = config.pretrain_model(path_prefix=path_prefix)

        # 2. Get labels for evaluation data.
        labels = config.get_labels(path_prefix=path_prefix)

        # 3. Get predictions of model before rlhf and record which ones are correct.
        preds_before = config.get_predictions(path_prefix=path_prefix, model=model)
        correct_preds_before = cls.get_correct_preds(preds=preds_before, labels=labels)
        acc_before_rlhf = np.mean(correct_preds_before)

        # 4. rlhf
        config.rlhf(path_prefix=path_prefix, model=model)

        # 5. Get predictions after rlhf and record which ones are correct.
        preds_after = config.get_predictions(path_prefix=path_prefix, model=model)
        correct_preds_after = cls.get_correct_preds(preds=preds_after, labels=labels)
        acc_after_rlhf = np.mean(correct_preds_after)

        # 6. Calculate how many of the originally correct predictions are still correct.
        num_correct_before = np.sum(correct_preds_before)
        num_still_correct = np.sum(correct_preds_before * correct_preds_after)

        # 7. Calculate how many of the originally incorrect predictions are now correct.
        incorrect_preds_before = 1 - correct_preds_before
        num_incorrect_before = np.sum(incorrect_preds_before)
        num_improved = np.sum(incorrect_preds_before * correct_preds_after)

        # 8. Report metrics.
        acc_on_correct = num_still_correct / num_correct_before
        acc_on_incorrect = num_improved / num_incorrect_before

        metrics = {
            "p_at_1_before_rlhf": acc_before_rlhf,
            "p_at_1_after_rlhf": acc_after_rlhf,
            "p_at_1_on_correct": acc_on_correct,
            "p_at_1_on_incorrect": acc_on_incorrect,
        }
        print(metrics)

        if mlflow_logger:
            mlflow.log_metrics(metrics)

        config.cleanup()

    @staticmethod
    def get_correct_preds(preds, labels):
        # Returns a np array of 0/1 values indicating which predictions are correct.
        assert len(preds) == len(labels)

        correct_preds = np.zeros(len(preds))
        for i, (pred, label) in enumerate(zip(preds, labels)):
            if pred in label:
                correct_preds[i] = 1

        return correct_preds
