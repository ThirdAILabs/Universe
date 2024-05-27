# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar, Union, final

import pandas as pd
from langchain_community.document_loaders import (
    UnstructuredEmailLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
)

try:
    from unstructured.cleaners.core import (
        clean_bullets,
        clean_extra_whitespace,
        clean_ligatures,
        clean_non_ascii_chars,
        replace_mime_encodings,
        replace_unicode_quotes,
    )
except Exception as e:
    raise ModuleNotFoundError(
        "To use NeuralDB with these document types please run: pip3 install unstructured[all-docs]"
    )


from .utils import (
    chunk_text,
    clean_text,
    clean_text_and_remove_urls,
    ensure_valid_encoding,
)

PPTX_CHUNK_THRESHOLD: final = 30


@dataclass
class UnstructuredParagraph:
    para: str
    filename: str
    filetype: str
    page: Optional[int]
    display: str


@dataclass
class EmlParagraph(UnstructuredParagraph):
    subject: str
    sent_from: str
    sent_to: str


class UnstructuredParse:
    def __init__(self, filepath: str):
        self._filepath = filepath
        self._filename = str(Path(filepath).name)
        self._ext = Path(filepath).suffix[1:]  # Removing '.' from the extension
        self._post_processors = [
            clean_extra_whitespace,
            clean_non_ascii_chars,
            clean_bullets,
            clean_ligatures,
            replace_unicode_quotes,
            replace_mime_encodings,
        ]
        self._error_msg = f"Cannot process {self._ext} file: {self._filepath}"

    def process_elements(
        self,
    ) -> Tuple[Union[List[Type[UnstructuredParagraph]], str], bool]:
        raise NotImplementedError()

    def create_train_df(
        self, paragraphs: List[Type[UnstructuredParagraph]]
    ) -> pd.DataFrame:
        columns = paragraphs[0].__dict__.keys()
        df = pd.DataFrame(index=range(len(paragraphs)), columns=columns)

        for i, elem in enumerate(paragraphs):
            df.iloc[i] = elem.__dict__

        for column in ["para", "display"]:
            df[column] = df[column].apply(ensure_valid_encoding)
        return df


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
            print(str(e))
            print(self._error_msg)

    def process_elements(
        self,
    ) -> Tuple[Union[List[Type[UnstructuredParagraph]], str], bool]:
        paragraphs = []
        try:
            docs = self.PptxLoader.load()
            current_text = ""
            last_page_no = len(docs)
            for doc in docs:
                text = clean_text(text=current_text + " " + doc.page_content)
                chunks = chunk_text(text)
                if len(chunks[-1]) < PPTX_CHUNK_THRESHOLD:
                    if len(chunks) == 1:
                        if last_page_no != doc.metadata["page_number"]:
                            current_text = text
                            continue
                        elif len(paragraphs) > 0:
                            paragraphs[-1].para += " " + text
                            continue
                    else:
                        chunks[-2] += " " + chunks[-1]
                        chunks.pop()

                rows = [
                    UnstructuredParagraph(
                        para=chunk,
                        filename=self._filename,
                        filetype=self._ext,
                        page=doc.metadata["page_number"],
                        display=str(chunk.replace("\n", " ")),
                    )
                    for chunk in chunks
                ]
                paragraphs.extend(rows)

            return (
                (paragraphs, True)
                if len(paragraphs) > 0
                else (f"Empty pptx file OR {self._error_msg}", False)
            )
        except Exception as e:
            print(str(e))
            return self._error_msg, False


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
            print(str(e))
            print(self._error_msg)

    def process_elements(
        self,
    ) -> Tuple[Union[List[Type[UnstructuredParagraph]], str], bool]:
        try:
            docs = self.EmlLoader.load()
            text = ""
            for doc in docs:
                content = doc.page_content
                text += clean_text_and_remove_urls(content) + " "
            text = re.sub(pattern=r"\s+", repl=" ", string=text).strip()

            paragraphs = [
                EmlParagraph(
                    para=chunk,
                    filename=self._filename,
                    filetype=self._ext,
                    page=None,
                    display=chunk,
                    subject=doc.metadata["subject"],
                    sent_from=",".join(doc.metadata["sent_from"]),
                    sent_to=",".join(doc.metadata["sent_to"]),
                )
                for chunk in chunk_text(text)
            ]

            return (
                (paragraphs, True)
                if len(paragraphs) > 0
                else (f"Empty eml file OR {self._error_msg}", False)
            )
        except Exception as e:
            print(str(e))
            return self._error_msg, False


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
            print(self._error_msg)

    def process_elements(
        self,
    ) -> Tuple[Union[List[Type[UnstructuredParagraph]], str], bool]:
        try:
            doc = self.TxtLoader.load()
            content = clean_text(doc[0].page_content)

            paragraphs = [
                UnstructuredParagraph(
                    para=chunk,
                    filename=self._filename,
                    filetype=self._ext,
                    page=None,
                    display=chunk,
                )
                for chunk in chunk_text(content)
            ]
            return (
                (paragraphs, True)
                if len(paragraphs) > 0
                else (f"Empty txt file OR {self._error_msg}", False)
            )
        except Exception as e:
            print(str(e))
            return self._error_msg, False
