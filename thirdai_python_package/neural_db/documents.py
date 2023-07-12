import hashlib
import os
import shutil
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from pytrie import StringTrie
from requests.models import Response
from thirdai.dataset.data_source import PyDataSource

from .parsing_utils import doc_parse, pdf_parse, url_parse
from .utils import hash_file, hash_string


class Reference:
    pass


class Document:
    @property
    def size(self) -> int:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def reference(self, element_id: int) -> Reference:
        raise NotImplementedError()

    @property
    def hash(self) -> str:
        sha1 = hashlib.sha1()
        sha1.update(bytes(self.name, "utf-8"))
        for i in range(self.size):
            sha1.update(bytes(self.reference(i).text, "utf-8"))
        return sha1.hexdigest()

    def strong_text(self, element_id: int) -> str:
        return self.reference(element_id).text

    def weak_text(self, element_id: int) -> str:
        return self.reference(element_id).text

    def context(self, element_id: int, radius: int) -> str:
        window_start = max(0, element_id - radius)
        window_end = min(self.size, element_id + radius + 1)
        return " \n".join(
            [self.reference(elid).text for elid in range(window_start, window_end)]
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass


class Reference:
    def __init__(
        self,
        document: Document,
        element_id: int,
        text: str,
        source: str,
        metadata: dict,
        upvote_ids: List[int] = None,
    ):
        self._id = element_id
        self._upvote_ids = upvote_ids if upvote_ids is not None else [element_id]
        self._text = text
        self._source = source
        self._metadata = metadata
        self._context_fn = lambda radius: document.context(element_id, radius)

    @property
    def id(self):
        return self._id

    @property
    def upvote_ids(self):
        return self._upvote_ids

    @property
    def text(self):
        return self._text

    @property
    def source(self):
        return self._source

    @property
    def metadata(self):
        return self._metadata

    def context(self, radius: int):
        return self._context_fn(radius)


class DocumentRow:
    def __init__(self, element_id: int, strong: str, weak: str):
        self.id = element_id
        self.strong = strong
        self.weak = weak


class DocumentDataSource(PyDataSource):
    def __init__(self, id_column, strong_column, weak_column):
        PyDataSource.__init__(self)
        self.documents: List[Tuple[Document, int]] = []
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column
        self._size = 0
        self.restart()

    def add(self, document: Document, start_id: int):
        self.documents.append((document, start_id))
        self._size += document.size

    def row_iterator(self):
        for doc, start_id in self.documents:
            for i in range(doc.size):
                yield DocumentRow(
                    element_id=start_id + i,
                    strong=doc.strong_text(i),
                    weak=doc.weak_text(i),
                )

    @property
    def size(self):
        return self._size

    def _csv_line(self, element_id: str, strong: str, weak: str):
        df = pd.DataFrame(
            {
                self.id_column: [element_id],
                self.strong_column: [strong],
                self.weak_column: [weak],
            }
        )
        return df.to_csv(header=None, index=None).strip("\n")

    def _get_line_iterator(self):
        # First yield the header
        yield self._csv_line(self.id_column, self.strong_column, self.weak_column)
        # Then yield rows
        for row in self.row_iterator():
            yield self._csv_line(element_id=row.id, strong=row.strong, weak=row.weak)

    def resource_name(self) -> str:
        return "Documents:\n" + "\n".join([doc.name for doc, _ in self.documents])


class IntroAndTrainDocuments:
    def __init__(self, intro: DocumentDataSource, train: DocumentDataSource) -> None:
        self.intro = intro
        self.train = train


class DocumentManager:
    def __init__(self, id_column, strong_column, weak_column) -> None:
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column
        self.registry: Dict[str, Tuple[Document, int]] = {}
        self.id_sorted_docs: List[Tuple[Document, int]] = []
        self.source_id_prefix_trie = StringTrie()

    def _next_id(self):
        if len(self.id_sorted_docs) == 0:
            return 0
        doc, start_id = self.id_sorted_docs[-1]
        return start_id + doc.size

    def add(self, documents: List[Document]):
        intro = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        train = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        for doc in documents:
            doc_hash = doc.hash
            if doc_hash not in self.registry:
                start_id = self._next_id()
                # Adding this tuple to two data structures does not double the
                # memory usage because Python uses references.
                doc_and_id = (doc, start_id)
                self.registry[doc_hash] = doc_and_id
                self.id_sorted_docs.append(doc_and_id)
                self.source_id_prefix_trie[doc_hash] = doc_hash
                intro.add(doc, start_id)
            doc, start_id = self.registry[doc_hash]
            train.add(doc, start_id)

        return IntroAndTrainDocuments(intro=intro, train=train), [
            doc.hash for doc in documents
        ]

    def sources(self):
        return {doc_hash: doc.name for doc_hash, (doc, _) in self.registry.items()}

    def match_source_id_by_prefix(self, prefix: str) -> Document:
        if prefix in self.registry:
            return [prefix]
        return self.source_id_prefix_trie.values(prefix)

    def source_by_id(self, source_id: str):
        return self.registry[source_id]

    def clear(self):
        self.registry = {}
        self.id_sorted_docs = []
        self.source_id_prefix_trie = StringTrie()

    def _get_doc_and_start_id(self, element_id: int):
        # Iterate through docs in reverse order
        for i in range(len(self.id_sorted_docs) - 1, -1, -1):
            doc, start_id = self.id_sorted_docs[i]
            if start_id <= element_id:
                return self.id_sorted_docs[i]
        raise ValueError(f"Unable to find document that has id {id}.")

    def reference(self, element_id: int):
        doc, start_id = self._get_doc_and_start_id(element_id)
        doc_ref = doc.reference(element_id - start_id)
        doc_ref._id = element_id
        doc_ref._upvote_ids = [start_id + uid for uid in doc_ref._upvote_ids]
        return doc_ref

    def context(self, element_id: int, radius: int):
        doc, start_id = self._get_doc_and_start_id(element_id)
        return doc.context(element_id - start_id, radius)

    def get_data_source(self) -> DocumentDataSource:
        data_source = DocumentDataSource(
            id_column=self.id_column,
            strong_column=self.strong_column,
            weak_column=self.weak_column,
        )

        for doc, start_id in self.id_sorted_docs:
            data_source.add(document=doc, start_id=start_id)

        return data_source

    def save_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.id_sorted_docs):
            subdir = directory / str(i)
            os.mkdir(subdir)
            doc.save_meta(subdir)

    def load_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.id_sorted_docs):
            subdir = directory / str(i)
            doc.load_meta(subdir)


class CSV(Document):
    def __init__(
        self, path, id_column, strong_columns, weak_columns, reference_columns
    ) -> None:
        self.df = pd.read_csv(path)
        self.df = self.df.sort_values(id_column)
        assert len(self.df[id_column].unique()) == len(self.df[id_column])
        assert self.df[id_column].min() == 0
        assert self.df[id_column].max() == len(self.df[id_column]) - 1

        for col in strong_columns + weak_columns:
            self.df[col] = self.df[col].fillna("")

        self.path = Path(path)
        self._hash = hash_file(path)
        self.id_column = id_column
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = reference_columns

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def name(self) -> str:
        return self.path.name

    def strong_text(self, element_id: int) -> str:
        row = self.df.iloc[element_id]
        return " ".join([row[col] for col in self.strong_columns])

    def weak_text(self, element_id: int) -> str:
        row = self.df.iloc[element_id]
        return " ".join([row[col] for col in self.weak_columns])

    def reference(self, element_id: int) -> Reference:
        row = self.df.iloc[element_id]
        text = " ".join([row[col] for col in self.reference_columns])
        return Reference(
            document=self,
            element_id=element_id,
            text=text,
            source=str(self.path.absolute()),
            metadata=row.to_dict(),
        )

    def context(self, element_id: int, radius) -> str:
        rows = self.df.iloc[
            max(0, element_id - radius) : min(len(self.df), element_id + radius + 1)
        ]
        return " ".join([row[col] for col in self.reference_columns for row in rows])

    def save_meta(self, directory: Path):
        # Let's copy the original CSV file to the provided directory
        shutil.copy(self.path, directory)

    def load_meta(self, directory: Path):
        # Since we've moved the CSV file to the provided directory, let's make
        # sure that we point to this CSV file.
        self.path = directory / self.path.name


# Base class for PDF and DOCX classes because they share the same logic.
class Extracted(Document):
    def __init__(
        self,
        filename: str,
    ):
        self.filename = Path(filename)
        self.df = self.process_data(filename)
        self.hash_val = hash_file(filename)

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def hash(self) -> str:
        return self.hash_val

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def name(self) -> str:
        return self.filename.name

    def strong_text(self, element_id: int) -> str:
        return self.df["passage"].iloc[element_id]

    def weak_text(self, element_id: int) -> str:
        return self.df["para"].iloc[element_id]

    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        return Reference(
            document=self,
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=str(self.filename.absolute()),
            metadata=self.df.iloc[element_id].to_dict(),
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")

        rows = self.df.iloc[
            max(0, element_id - radius) : min(len(self.df), element_id + radius + 1)
        ]
        return "\n".join(rows["passage"])

    def save_meta(self, directory: Path):
        # Let's copy the original file to the provided directory
        shutil.copy(self.filename, directory)

    def load_meta(self, directory: Path):
        # Since we've moved the file to the provided directory, let's make
        # sure that we point to this file.
        self.filename = directory / self.filename.name


def process_pdf(filename: str) -> pd.DataFrame:
    elements, success = pdf_parse.process_pdf_file(filename)

    if not success:
        raise ValueError(f"Could not read PDF file: {filename}")

    elements_df = pdf_parse.create_train_df(elements)

    return elements_df


def process_docx(filename: str) -> pd.DataFrame:
    elements, success = doc_parse.get_elements(filename)

    if not success:
        raise ValueError(f"Could not read DOCX file: {filename}")

    elements_df = doc_parse.create_train_df(elements)

    return elements_df


class PDF(Extracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(filename=filename)

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        return process_pdf(filename)


class DOCX(Extracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(filename=filename)

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        return process_docx(filename)


class URL(Document):
    def __init__(self, url: str, url_response: Response = None):
        self.url = url
        self.df = self.process_data(url, url_response)
        self.hash_val = hash_string(url)

    def process_data(self, url, url_response=None) -> pd.DataFrame:
        # Extract elements from each file
        elements, success = url_parse.process_url(url, url_response)

        if not success or not elements:
            raise ValueError(f"Could not retrieve data from URL: {url}")

        elements_df = url_parse.create_train_df(elements)

        return elements_df

    def hash(self) -> str:
        return self.hash_val

    def size(self) -> int:
        return len(self.df)

    def name(self) -> str:
        return self.url

    def strong_text(self, element_id: int) -> str:
        return self.df["text"].iloc[element_id]

    def weak_text(self, element_id: int) -> str:
        return self.df["text"].iloc[element_id]

    def reference(self, element_id: int) -> Reference:
        return Reference(
            document=self,
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=self.url,
            metadata={},
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")
        rows = self.df.iloc[
            max(0, element_id - radius) : min(len(self.df), element_id + radius + 1)
        ]
        return "\n".join(rows["text"])


class SentenceLevelExtracted(Extracted):
    """Parses a document into sentences and creates a NeuralDB entry for each
    sentence. The strong column of the entry is the sentence itself while the
    weak column is the paragraph from which the sentence came. A NeuralDB
    reference produced by this object displays the paragraph instead of the
    sentence to increase recall.
    """

    def __init__(
        self,
        filename: str,
    ):
        self.filename = Path(filename)
        self.df = self.parse_sentences(self.process_data(filename))
        self.hash_val = hash_file(filename)
        self.para_df = self.df["para"].unique()

    def not_just_punctuation(sentence: str):
        for character in sentence:
            if character not in string.punctuation and not character.isspace():
                return True
        return False

    def get_sentences(paragraph: str):
        return [
            sentence
            for sentence in sent_tokenize(paragraph)
            if SentenceLevelExtracted.not_just_punctuation(sentence)
        ]

    def parse_sentences(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df["sentences"] = df["passage"].apply(SentenceLevelExtracted.get_sentences)
        df.drop("passage", axis=1, inplace=True)

        num_sents_cum_sum = np.cumsum(df["sentences"].apply(lambda sents: len(sents)))
        df["id_offsets"] = np.zeros(len(df))
        df["id_offsets"][1:] = num_sents_cum_sum[:-1]
        df["id_offsets"] = df["id_offsets"].astype(int)

        def get_ids(record):
            id_offset = record["id_offsets"]
            n_sents = len(record["sentences"])
            return list(range(id_offset, id_offset + n_sents))

        df = pd.DataFrame.from_records(
            [
                {
                    "passage": sentence,
                    "para_id": para_id,
                    "sentence_id": i + record["id_offsets"],
                    "sentence_ids_in_para": get_ids(record),
                    **record,
                }
                for para_id, record in enumerate(df.to_dict(orient="records"))
                for i, sentence in enumerate(record["sentences"])
            ]
        )

        df.drop("sentences", axis=1, inplace=True)
        df.drop("id_offsets", axis=1, inplace=True)
        return df

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def hash(self) -> str:
        return self.hash_val

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def name(self) -> str:
        return self.filename.name

    def strong_text(self, element_id: int) -> str:
        return self.df["passage"].iloc[element_id]

    def weak_text(self, element_id: int) -> str:
        return self.df["para"].iloc[element_id]

    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        return Reference(
            document=self,
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=str(self.filename.absolute()),
            metadata=self.df.iloc[element_id].to_dict(),
            upvote_ids=self.df["sentence_ids_in_para"].iloc[element_id],
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")

        para_id = self.df.iloc[element_id]["para_id"]

        rows = self.para_df[
            max(0, para_id - radius) : min(len(self.para_df), para_id + radius + 1)
        ]
        return "\n\n".join(rows)

    def save_meta(self, directory: Path):
        # Let's copy the original file to the provided directory
        shutil.copy(self.filename, directory)

    def load_meta(self, directory: Path):
        # Since we've moved the file to the provided directory, let's make
        # sure that we point to this file.
        self.filename = directory / self.filename.name


class SentenceLevelPDF(SentenceLevelExtracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(filename=filename)

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        return process_pdf(filename)


class SentenceLevelDOCX(SentenceLevelExtracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(filename=filename)

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        return process_docx(filename)
