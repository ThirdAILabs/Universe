import json

from thirdai.dataset.data_source import PyDataSource


def tokenize_text(tokenizer, text):
    tokens = tokenizer.encode(text)
    return " ".join(map(str, tokens))


class NerDataSource(PyDataSource):
    def __init__(self, file_path=None, pretrained=False):
        self.pretrained = pretrained
        if file_path:
            self.file_path = file_path
        if self.pretrained:
            try:
                from transformers import GPT2Tokenizer

                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            except ImportError:
                raise ImportError(
                    "transformers library is not installed. Please install it to use LLMDataSource."
                )
        PyDataSource.__init__(self)

        self.restart()

    def _get_line_iterator(self):
        with open(self.file_path, "r") as file:
            for line in file:
                json_obj = json.loads(line.strip())
                if self.pretrained:
                    json_obj["source"] = [
                        tokenize_text(self.tokenizer, token)
                        for token in json_obj["source"]
                    ]
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
