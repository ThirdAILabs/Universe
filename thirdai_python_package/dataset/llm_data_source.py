import json

from thirdai.dataset.data_source import PyDataSource


def tokenize_and_dump_json(tokenizer, json_obj):
    json_obj["target"] = tokenize_text(tokenizer, json_obj["target"])
    if "context" in json_obj:
        json_obj["context"] = tokenize_text(tokenizer, json_obj["context"])
    if "prompt" in json_obj:
        json_obj["prompt"] = tokenize_text(tokenizer, json_obj["prompt"])
    return json.dumps(json_obj)


def tokenize_text(tokenizer, text):
    tokens = tokenizer.encode(text)
    return " ".join(map(str, tokens))


class LLMDataSource(PyDataSource):
    def __init__(self, file_path):
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
                tokenized_json_obj = tokenize_and_dump_json(self.tokenizer, json_obj)
                yield tokenized_json_obj

    def resource_name(self) -> str:
        return self.file_path


class RayTextDataSource(PyDataSource):
    def __init__(self, ray_dataset, should_tokenize=False):
        PyDataSource.__init__(self)
        self.ray_dataset = ray_dataset
        self.should_tokenize = should_tokenize
        try:
            import ray
            from transformers import GPT2Tokenizer
        except ImportError:
            raise ImportError(
                "This class requires both the 'ray' and 'transformers' libraries. Please ensure they are installed."
            )
        if self.should_tokenize:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.restart()

    def _get_line_iterator(self):
        for row in self.ray_dataset.iter_rows():
            text = row["text"]
            if self.should_tokenize:
                json_obj = json.loads(text.strip())
                text = tokenize_and_dump_json(self.tokenizer, json_obj)
            yield text

    def resource_name(self) -> str:
        return f"ray-dataset-sources"
