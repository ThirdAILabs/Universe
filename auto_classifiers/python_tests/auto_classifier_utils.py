def compute_accuracy_of_predictions(test_labels, predictions):
    assert len(predictions) == len(test_labels)
    return sum(
        (prediction == answer) for (prediction, answer) in zip(predictions, test_labels)
    ) / len(predictions)


# This function checks that the the results of predict_single and predict batch
# match the original predictions.
def check_autoclassifier_predict_correctness(
    classifier, test_samples, original_predictions
):
    for sample, original_prediction in zip(test_samples, original_predictions):
        single_prediction = classifier.predict(sample)
        assert single_prediction == original_prediction

    for samples, predictions in batch_predictions(test_samples, original_predictions):
        batched_prediction = classifier.predict_batch(samples)
        for prediction, original_prediction in zip(batched_prediction, predictions):
            assert prediction == original_prediction


def batch_predictions(original_predictions, samples, batch_size=10):
    batches = []
    for i in range(0, len(original_predictions), batch_size):
        batches.append(
            (original_predictions[i : i + batch_size], samples[i : i + batch_size])
        )
    return batches
