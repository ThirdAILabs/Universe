import json

from thirdai.dataset.data_source import PyDataSource


def tokenize_text(tokenizer, text):
    tokens = tokenizer.encode(text)
    return " ".join(map(str, tokens))


class NerDataSource(PyDataSource):
    def __init__(self, type, file_path=None, token_column=None, tag_column=None):
        PyDataSource.__init__(self)

        self.file_path = file_path
        self.token_column = token_column
        self.tag_column = tag_column
        self.pretrained = type == "bolt_ner"

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
        with open(self.file_path, "r") as file:
            for line in file:

                json_obj = json.loads(line.strip())
                if not all(
                    column in json_obj
                    for column in [self.tag_column, self.token_column]
                ):
                    raise ValueError(
                        f"{self.tag_column} or {self.token_column} doesn't exist in the column, line: {line}"
                    )
                if self.pretrained:
                    json_obj["tokens"] = [
                        tokenize_text(self.tokenizer, token)
                        for token in json_obj[self.token_column]
                    ]
                else:
                    json_obj["tokens"] = json_obj[self.token_column]

                json_obj["tags"] = json_obj[self.tag_column]
                data = json.dumps(json_obj)

                yield data

    def inference_featurizer(self, sentence_tokens_list):
        if self.pretrained:
            return [
                [tokenize_text(self.tokenizer, token) for token in sentence_tokens]
                for sentence_tokens in sentence_tokens_list
            ]
        return sentence_tokens_list

    def resource_name(self) -> str:
        return self.file_path
