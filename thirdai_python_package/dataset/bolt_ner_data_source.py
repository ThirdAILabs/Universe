import json

from thirdai.dataset.data_source import PyDataSource


def tokenize_text(tokenizer, text):
    tokens = tokenizer.encode(text)
    return " ".join(map(str, tokens))


class NerBoltDataSource(PyDataSource):
    def __init__(self, file_path=None):
        if file_path:
            self.file_path = file_path
        try:
            from transformers import GPT2Tokenizer
        except ImportError:
            raise ImportError(
                "transformers library is not installed. Please install it to use LLMDataSource."
            )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        PyDataSource.__init__(self)

        self.restart()

    def _get_line_iterator(self):
        with open(self.file_path, "r") as file:
            for line in file:
                json_obj = json.loads(line.strip())
                json_obj["source"] = [
                    tokenize_text(self.tokenizer, token) for token in json_obj["source"]
                ]
                data = json.dumps(json_obj)
                yield data

    def inference_featurizer(self, sentence_tokens_list):
        return [
            [tokenize_text(self.tokenizer, token) for token in sentence_tokens]
            for sentence_tokens in sentence_tokens_list
        ]

    def resource_name(self) -> str:
        return self.file_path


class NerDataSource(PyDataSource):
    def __init__(self, file_path=None):
        PyDataSource.__init__(self)
        if file_path:
            self.file_path = file_path
        self.restart()

    def _get_line_iterator(self):
        with open(self.file_path, "r") as file:
            for line in file:
                json_obj = json.loads(line.strip())
                data = json.dumps(json_obj)
                yield data

    def inference_featurizer(self, sentence_tokens_list):
        return sentence_tokens_list

    def resource_name(self) -> str:
        return self.file_path
