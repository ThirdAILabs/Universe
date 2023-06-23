import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from requests.models import Response

import pandas as pd
from parsing_utils import doc_parse, pdf_parse, url_parse
from thirdai.neural_db.qa import ContextArgs
from thirdai.neural_db.utils import hash_string, hash_file
from thirdai.neural_db import Document, Reference




# Base class for PDF and DOCX classes because they share the same logic.
class Extracted(Document):
    def __init__(
        self,
        filename: str,
    ):
        
        self.filename = filename
        self.df = self.process_data(filename)
        self.hash_val = hash_file(filename)

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def hash(self) -> str:
        return self.hash_val

    def size(self) -> int:
        return len(self.df)

    def name(self) -> str:
        return self.filename

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
            source=self.filename,
            metadata={"page": self.df["page"].iloc[element_id]},
        )

    def context(self, element_id: int, radius: int) -> str:
        if not 0 <= element_id < self.size():
            raise ("Element id not in document.")
        
        window_start = max(0, element_id - radius)
        window_end = min(self.size(), element_id + radius + 1)
        window_chunks = self.df.iloc[window_start:window_end]["passage"].tolist()
        return "\n".join(window_chunks)


class PDF(Extracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(
            filename=filename
        )

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        elements, success = pdf_parse.process_pdf_file(filename)

        if not success:
            print(f"Could not read PDF file {filename}")
            return pd.DataFrame()

        elements_df = pdf_parse.create_train_df(elements)

        return elements_df


class DOCX(Extracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(
            filename=filename
        )

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        elements, success = doc_parse.get_elements(filename)

        if not success:
            print(f"Could not read DOCX file {filename}")
            return pd.DataFrame()

        elements_df = doc_parse.create_train_df(elements)

        return elements_df


class URL(Document):
    def __init__(
        self,
        url: str,
        url_response: Response = None
    ):
        self.url = url
        self.df = self.process_data(url, url_response)
        self.hash_val = hash_string(url)

    def process_data(self, url, url_response=None) -> pd.DataFrame:
        # Extract elements from each file
        elements, success = url_parse.process_url(url, url_response)

        if not success or not elements:
            return pd.DataFrame()

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
    
    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        return Reference(
            document=self,
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=self.url,
            metadata={},
        )

    def context(self, element_id, context_args) -> str:
        if not 0 <= element_id < self.size():
            raise ("Element id not in document.")

        if hasattr(context_args, "chunk_radius"):
            chunk_radius = context_args.chunk_radius

            center_chunk_idx = element_id

            window_start = center_chunk_idx - chunk_radius // 2
            window_end = window_start + chunk_radius

            if window_start < 0:
                window_start = 0
                window_end = min(chunk_radius, self.size())
            elif window_end > self.size():
                window_end = self.size()
                window_start = max(0, self.size() - chunk_radius)

            window_chunks = self.df.iloc[window_start:window_end]["text"].tolist()
            return "\n".join(window_chunks)

        return ""


class CSV(Document):
    def __init__(self, filename, strong_cols, weak_cols):
        self.filename = filename
        self.strong_cols = strong_cols
        self.weak_cols = weak_cols
        self.df = pd.read_csv(self.filename)
        self.hash_val = hash_file(filename)

        for col in strong_cols + weak_cols:
            self.df[col] = self.df[col].fillna("")

    def hash(self) -> str:
        return self.hash_val

    def size(self) -> int:
        return len(self.df)

    def name(self) -> str:
        return self.filename

    def strong_text(self, element_id: int) -> str:
        return " ".join(self.df[self.strong_cols].iloc[element_id].tolist())
       
    def weak_text(self, element_id: int) -> str:
        return " ".join(self.df[self.weak_cols].iloc[element_id].tolist())
    
    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        return Reference(
            document=self,
            element_id=element_id,
            text=self.weak_text(element_id),
            source=self.filename,
            metadata={},
        )
