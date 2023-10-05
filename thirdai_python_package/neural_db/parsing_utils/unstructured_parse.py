# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from langchain.document_loaders import (
    UnstructuredEmailLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from unstructured.cleaners.core import (
    clean_bullets,
    clean_extra_whitespace,
    clean_ligatures,
    clean_non_ascii_chars,
    clean_ordered_bullets,
    replace_mime_encodings,
    replace_unicode_quotes,
)

from utils import chunk_text, clean_text_and_remove_urls, ensure_valid_encoding


@dataclass
class UnstructuredParagraph:
    para: str
    filepath: str
    filetype: str
    page_no: int
    display: str


@dataclass
class EmlParagraph(UnstructuredParagraph):
    subject: str
    sent_from: str
    sent_to: str


class UnstructuredParse:
    def __init__(self, filepath: str):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75,
            length_function=len,
        )
        self._filepath = filepath
        self._ext = Path(filepath).suffix
        self._post_processors = [
            clean_extra_whitespace,
            clean_non_ascii_chars,
            clean_bullets,
            clean_ordered_bullets,
            clean_ligatures,
            replace_unicode_quotes,
            replace_mime_encodings,
        ]

    def process_elements(self) -> Tuple[Union[UnstructuredParagraph, str], bool]:
        raise NotImplementedError()

    def create_train_df(self, paragraphs: UnstructuredParagraph) -> pd.DataFrame:
        raise NotImplementedError()


class PptxParse(UnstructuredParse):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        try:
            self.PptxLoader = UnstructuredPowerPointLoader(
                file_path=self._filepath,
                mode="paged",
                post_processors=self._post_processors,
            )
        except Exception as e:
            print(e.__str__())
            print("Cannot process file:", filepath)

    def process_elements(self) -> Tuple[Union[UnstructuredParagraph, str], bool]:
        paragraphs = []
        try:
            docs = self.PptxLoader.load()
            for doc in docs:
                text = doc.page_content
                text = (
                    str(text)
                    .strip()
                    .replace("\r\n", " ")
                    .replace("\n", " ")
                    .replace("\t", " ")
                    .lower()
                )
                chunks = chunk_text(text)

                row = [
                    UnstructuredParagraph(
                        para=chunk,
                        filepath=self._filepath,
                        filetype=self._ext,
                        page_no=doc.metadata["page_number"],
                        display=str(text.replace("\n", " ")),
                    )
                    for chunk in chunks
                ]
                paragraphs.extend(row)

            return paragraphs, True
        except Exception as e:
            print(e.__str__())
            return "Cannot process pptx file: " + self._filepath, False

    def create_train_df(self, paragraphs: List[UnstructuredParagraph]) -> pd.DataFrame:
        df = pd.DataFrame(
            index=range(len(paragraphs)),
            columns=["para", "filename", "filetype", "page", "display"],
        )
        for i, elem in enumerate(paragraphs):
            df.iloc[i] = [
                elem.para,
                elem.filepath,
                elem.filetype,
                elem.page_no,
                elem.display,
            ]

        for column in ["para", "display"]:
            df[column] = df[column].apply(ensure_valid_encoding)
        return df


class EmlParse(UnstructuredParse):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        try:
            self.EmlLoader = UnstructuredEmailLoader(
                file_path=self._filepath,
                mode="elements",
                post_processors=self._post_processors,
            )
        except Exception as e:
            print(e.__str__())
            print("Cannot process file:", filepath)

    def process_elements(self) -> Tuple[Union[EmlParagraph, str], bool]:
        rows = []
        try:
            docs = self.EmlLoader.load()
            text = ""
            for doc in docs:
                content = doc.page_content
                text += clean_text_and_remove_urls(content).lower() + " "
            text = re.sub(pattern=r"\s+", repl=" ", string=text)
            chunks = chunk_text(text)

            row = [
                EmlParagraph(
                    para=chunk,
                    filepath=self._filepath,
                    filetype=self._ext,
                    page_no=None,
                    display=str(text.replace("\n", " ")),
                    subject=doc.metadata["subject"],
                    sent_from=doc.metadata["sent_from"],
                    sent_to=doc.metadata["sent_to"],
                )
                for chunk in chunks
            ]
            rows.append(row)

            return rows, True
        except Exception as e:
            print(e.__str__())
            return "Cannot process Eml file: " + self._filepath, False

    def create_train_df(self, paragraphs: List[EmlParagraph]) -> pd.DataFrame:
        df = pd.DataFrame(
            index=range(len(paragraphs)),
            columns=[
                "para",
                "filename",
                "filetype",
                "page",
                "display",
                "subject",
                "sent_from",
                "sent_to",
            ],
        )
        for i, elem in enumerate(paragraphs):
            df.iloc[i] = [
                elem.para,
                elem.filepath,
                elem.filetype,
                elem.page_no,
                elem.display,
                elem.subject,
                elem.sent_from,
                elem.sent_to,
            ]

        for column in ["para", "display", "subject", "sent_from", "sent_to"]:
            df[column] = df[column].apply(ensure_valid_encoding)
        return df


class TxtParse(UnstructuredParse):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        try:
            self.TxtLoader = UnstructuredFileLoader(
                file_path=self._filepath,
                mode="single",
                post_processors=self._post_processors,
            )
        except Exception as e:
            print(str(e))
            print("Cannot process file: ", filepath)

    def process_elements(self) -> Tuple[Union[UnstructuredParagraph, str], bool]:
        try:
            doc = self.TxtLoader.load()
            content = (
                str(doc[0].page_content)
                .strip()
                .replace("\r\n", " ")
                .replace("\n", " ")
                .replace("\t", " ")
            )
            chunks = chunk_text(content)

            paragraphs = [
                UnstructuredParagraph(
                    para=chunk,
                    filepath=self._filepath,
                    filetype=self._ext,
                    page_no=None,
                    display=chunk,
                )
                for chunk in chunks
            ]
            return paragraphs, True
        except Exception as e:
            print(str(e))
            return "Cannot process txt file: " + self._filepath, False

    def create_train_df(self, paragraphs: List[UnstructuredParagraph]) -> pd.DataFrame:
        df = pd.DataFrame(
            index=range(len(paragraphs)),
            columns=["para", "filepath", "filetype", "page", "display"],
        )

        for idx, paragraph in enumerate(paragraphs):
            sentences = sent_tokenize(paragraph.para)
            sentences = [
                sent.replace("\t", " ")
                .replace(",", " ")
                .replace("\n", " ")
                .strip()
                .lower()
                for sent in sentences
            ]

            para = " ".join(sentences)
            df.iloc[idx] = [
                para,
                paragraph.filepath,
                paragraph.filetype,
                paragraph.page_no,
                paragraph.display,
            ]
        for column in ["para", "display"]:
            df[column] = df[column].apply(ensure_valid_encoding)

        return df, True


class DocParse(UnstructuredParse):
    pass
