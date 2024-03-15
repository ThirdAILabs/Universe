import os


def bert_base_uncased(dirname="."):
    BERT_TAG = "bert-base-uncased"
    BERT_VOCAB_PATH = os.path.join(dirname, f"{BERT_TAG}.vocab")
    BERT_VOCAB_URL = f"https://huggingface.co/{BERT_TAG}/resolve/main/vocab.txt"

    if not os.path.exists(BERT_VOCAB_PATH):
        import urllib.request

        response = urllib.request.urlopen(BERT_VOCAB_URL)
        with open(BERT_VOCAB_PATH, "wb+") as bert_vocab_file:
            bert_vocab_file.write(response.read())

    return BERT_VOCAB_PATH
