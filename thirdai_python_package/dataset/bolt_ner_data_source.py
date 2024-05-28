import json

from thirdai.dataset.data_source import PyDataSource
from thirdai import data, dataset


def tokenize_text(tokenizer, texts):
    text_tokens = tokenizer(texts)["input_ids"]
    return [" ".join(map(str, tokens)) for tokens in text_tokens]


class NerDataSource(PyDataSource):
    def __init__(
        self,
        model_type,
        tokens_column=None,
        tags_column=None,
        file_path=None,
        tokenizer=None,
    ):
        PyDataSource.__init__(self)

        self.file_path = file_path

        self.tokens_column = tokens_column
        self.tags_column = tags_column
        self.pretrained = model_type == "bolt_ner"

        if self.pretrained:
            try:
                CONFIG = data.transformations.NerFeatureConfig(
                    True, True, True, True, True, True, True
                )
                target_word_tokenizers = [
                    dataset.NaiveSplitTokenizer(" "),
                ]
                self.transform = data.transformations.NerTokenizerUnigram(
                    tokens_column="text",
                    featurized_sentence_column="fsc",
                    target_column="target",
                    target_dim=10,
                    dyadic_num_intervals=2,
                    target_word_tokenizers=target_word_tokenizers,
                    feature_enhancement_config=CONFIG,
                )
                from transformers import GPT2Tokenizer

                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            except ImportError:
                raise ImportError(
                    "transformers library is not installed. Please install it to use LLMDataSource."
                )
        if tokenizer:
            self.tokenizer = tokenizer
        self.restart()

    def _get_line_iterator(self):

        if self.file_path is None:
            raise ValueError(
                "The file path is None. Please provide a valid file path to access the data source."
            )

        if self.tags_column is None or self.tokens_column is None:
            raise ValueError(
                "Cannot load a datasource with either 'tokens_column' or 'tags_column' set to None. Please provide valid column names."
            )

        with open(self.file_path, "r") as file:
            for line in file:

                json_obj = json.loads(line.strip())
                if not all(
                    column in json_obj
                    for column in [self.tags_column, self.tokens_column]
                ):
                    raise ValueError(
                        f"{self.tags_column} or {self.tokens_column} doesn't exist in the column, line: {line}"
                    )
                if self.pretrained:
                    # transformed_token = [
                    #     self.transform.process_token([token], 0)[2:]
                    #     for token in json_obj[self.tokens_column]
                    # ]
                    json_obj[self.tokens_column] = tokenize_text(
                        self.tokenizer, json_obj[self.tokens_column]
                    )

                data = json.dumps(json_obj)

                yield data

    def inference_featurizer(self, sentence_tokens_list):
        if self.pretrained:
            # transformed_tokens = [
            #     [
            #         self.transform.process_token([token], 0)[2:]
            #         for token in sentence_tokens
            #     ]
            #     for sentence_tokens in sentence_tokens_list
            # ]
            return [
                tokenize_text(self.tokenizer, tokens) for tokens in sentence_tokens_list
            ]
        return sentence_tokens_list

    def resource_name(self) -> str:
        return self.file_path
