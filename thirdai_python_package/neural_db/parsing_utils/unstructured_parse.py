# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
from langchain.document_loaders import UnstructuredFileLoader
from nltk.tokenize import sent_tokenize
import parsing_utils.utils as utils

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

class UnstructuredParse():
    @property
    def create_train_df(self) -> int:
        raise NotImplementedError()

    @property
    def process_file(self) -> str:
        raise NotImplementedError()
    
    
