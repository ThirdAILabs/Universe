import re

import unidecode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize, word_tokenize

ATTACH_N_WORD_THRESHOLD = 20
MIN_WORDS_PER_CHUNK = 50
CHUNK_THRESHOLD = 150
MAX_CHUNK_LEN = 750
MIN_CHUNK_LEN = 20


# Convert a string to a unicode string
def ensure_valid_encoding(text):
    return unidecode.unidecode(text.encode("utf-8", "replace").decode("utf-8"))


# Validates a given chunk based on some rules
def valid_chunk(chunk):
    # Check if chunk is large enough
    if len(chunk) < MIN_CHUNK_LEN:
        return False

    # Check if chunk contains enough alphabet characters
    if sum(c.isalpha() for c in chunk) < len(chunk) / 2:
        return False

    return True


# Split a chunk into smaller chunks if the original chunk size is greater than MAX_CHUNK_LEN
def split_large_chunk(chunk, max_len, text_splitter):
    if len(chunk) > max_len:
        return text_splitter.split_text(chunk)
    return [chunk]


def chunk_text(text: str):
    sentences = sent_tokenize(text)
    if len(sentences) == 1:
        return [text] if valid_chunk(text) else []

    words_per_sentence = [len(word_tokenize(sent)) for sent in sentences]
    if sum(words_per_sentence) < CHUNK_THRESHOLD:
        return [text] if valid_chunk(text) else []

    chunks = []
    cur_word_count = 0
    start_idx = 0

    for idx in range(len(sentences)):
        word_count = words_per_sentence[idx]
        if cur_word_count < MIN_WORDS_PER_CHUNK:
            cur_word_count += word_count
        else:
            chunks.append(" ".join(sentences[start_idx:idx]))
            start_idx = idx
            cur_word_count = word_count

    if start_idx != len(sentences):
        final_chunk = " ".join(sentences[start_idx : len(sentences)])
        if len(chunks) > 0 and cur_word_count < MIN_WORDS_PER_CHUNK:
            chunks[-1] += final_chunk
        else:
            chunks.append(final_chunk)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=0,
        length_function=len,
    )

    chunks = [
        sub_chunk
        for chunk in chunks
        for sub_chunk in split_large_chunk(chunk, MAX_CHUNK_LEN, text_splitter)
    ]
    chunks = [chunk for chunk in chunks if valid_chunk(chunk)]

    return chunks


def clean_text_and_remove_urls(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    return text


def clean_text(text: str) -> str:
    text = (
        str(text)
        .strip()
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .lower()
    )

    return text
