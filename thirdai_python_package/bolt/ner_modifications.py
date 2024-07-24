from typing import Dict, List, Optional, Tuple

import thirdai
import thirdai._thirdai.bolt as bolt
from thirdai.dataset.bolt_ner_data_source import NerDataSource


def modify_ner():
    original_train = bolt.NER.train
    original_get_tags = bolt.NER.get_ner_tags

    def wrapped_train(
        self,
        filename,
        learning_rate: float = 1e-3,
        epochs: int = 5,
        batch_size: Optional[int] = 2000,
        train_metrics: List[str] = ["loss"],
        validation_file: Optional[str] = None,
        val_metrics: List[str] = [],
    ):
        train_data_source = NerDataSource(
            model_type=self.type(),
            tokens_column=self.tokens_column(),
            tags_column=self.tags_column(),
            file_path=filename,
        )

        if validation_file:
            validation_data_source = NerDataSource(
                model_type=self.type(),
                tokens_column=self.tokens_column(),
                tags_column=self.tags_column(),
                file_path=validation_file,
            )
        else:
            validation_data_source = None

        return original_train(
            self,
            train_data=train_data_source,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            train_metrics=train_metrics,
            val_data=validation_data_source,
            val_metrics=val_metrics,
        )

    def wrapped_predict_batch(self, tokens: List[List[str]], top_k: int = 1):
        assert top_k > 0
        inference_source = NerDataSource(self.type())
        featurized_tokens = inference_source.inference_featurizer(tokens)
        return original_get_tags(self, featurized_tokens, top_k)

    def wrapped_predict(self, tokens: List[str], top_k: int = 1):
        assert top_k > 0
        inference_source = NerDataSource(self.type())
        featurized_tokens = inference_source.inference_featurizer([tokens])
        return original_get_tags(self, featurized_tokens, top_k)[0]

    delattr(bolt.NER, "get_ner_tags")
    delattr(bolt.NER, "train")

    bolt.NER.train = wrapped_train
    bolt.NER.predict_batch = wrapped_predict_batch
    bolt.NER.predict = wrapped_predict

    bolt.NER.train_on_data_source = original_train
