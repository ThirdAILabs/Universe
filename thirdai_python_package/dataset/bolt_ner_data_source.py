import json
import pandas as pd

from thirdai.dataset.data_source import PyDataSource


def tokenize_text(tokenizer, text):
    tokens = tokenizer.encode(text.strip())
    return " ".join(map(str, tokens))


class NerDataSource(PyDataSource):
    def __init__(
        self, model_type, tokens_column=None, tags_column=None, file_path=None
    ):
        PyDataSource.__init__(self)

        self.file_path = file_path

        self.tokens_column = tokens_column
        self.tags_column = tags_column
        self.pretrained = model_type == "bolt_ner"

        if self.pretrained:
            try:
                from transformers import GPT2Tokenizer

                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            except ImportError:
                raise ImportError(
                    "transformers library is not installed. Please install it to use LLMDataSource."
                )

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

        dataframe = pd.read_csv(self.file_path)

        yield f"{self.tokens_column},{self.tags_column}"

        for source, target in zip(
            dataframe[self.tokens_column], dataframe[self.tags_column]
        ):
            source = source.replace(",", "")
            # Preprocess source text if using a pretrained model
            if self.pretrained:
                source = tokenize_text(self.tokenizer, source)
                print(f"{source=}, {target=}")

            yield f"{source},{target}"

    def inference_featurizer(self, sentences):
        if self.pretrained:
            return [tokenize_text(self.tokenizer, sentence) for sentence in sentences]
        return sentences

    def resource_name(self) -> str:
        return self.file_path
