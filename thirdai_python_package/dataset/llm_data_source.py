from thirdai.dataset.data_source import PyDataSource
import json


class PretrainingTextDataSource(PyDataSource):
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            from transformers import GPT2Tokenizer
        except ImportError:
            raise ImportError(
                "transformers library is not installed. Please install it to use PretrainingTextDataSource."
            )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        PyDataSource.__init__(self)
        self.restart()

    def _get_line_iterator(self):
        with open(self.file_path, "r") as file:
            for line in file:
                tokenized_line = self._tokenize(line.strip())
                yield tokenized_line

    def _tokenize(self, text):
        tokens = self.tokenizer.encode(text)
        tokenized_text = " ".join(map(str, tokens))
        json_output = json.dumps({"target": tokenized_text})
        return json_output

    def resource_name(self) -> str:
        return self.file_path
