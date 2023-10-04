# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPowerPointLoader, UnstructuredEmailLoader
from unstructured.cleaners.core import (clean_extra_whitespace, clean_non_ascii_chars, clean_bullets, clean_ordered_bullets, clean_ligatures, replace_mime_encodings, replace_unicode_quotes)
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass
import parsing_utils.utils as utils
from typing import List, Tuple, Union
from utils import chunk_text

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

@dataclass
class UnstructuredParagraph():
    para: str
    filepath: str
    page_no: int
    display: str
    
@dataclass
class EmlParagraph(UnstructuredParagraph):
    subject: str
    sent_from: str
    sent_to: str

# def process_unstructured_file(filename, text_splitter):
#     try:
#         rows = []
#         loader = UnstructuredFileLoader(filename, mode="elements")
#         docs = loader.load()
#         content = docs[0].page_content
#         texts = text_splitter.split_text(content)
#         for text in texts:
#             text = str(text).strip().replace("\r\n", " ").replace("\n", " ").replace("\t", " ").replace(",", " ")
#             row = [text, filename]
#             rows.append(row)
                
#         return rows, True
#     except Exception as e:
#         print(e.__str__())
#         return "Cannot process pdf file:" + filename, False
    
# def create_train_df(elements, id_col):
#     count = 0
#     df = pd.DataFrame(index=range(len(elements)), columns=[id_col, "passage", "para", "filename", "page", "display"])
#     for i, elem in enumerate(elements):
#         sents = sent_tokenize(elem[0])
#         sents = list(map(lambda x: x.lower(), sents))
#         passage = " ".join(sents)
#         df.iloc[i] = [elem[-1], passage, passage, elem[1], 0, passage]
#         count = count + 1
#     for column in ["passage"]:
#         df[column] = df[column].apply(utils.ensure_valid_encoding)
#     return df
    

class UnstructuredParse():
    def __init__(self, filepath: str):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 75,
            length_function = len,
        )
        self._filepath = filepath
        self._post_processors = [clean_extra_whitespace, clean_non_ascii_chars, clean_bullets, clean_ordered_bullets, clean_ligatures, replace_unicode_quotes, replace_mime_encodings]
    
    def process_elements(self) ->  Tuple[Union[UnstructuredParagraph, str], bool]:
        raise NotImplementedError()
    
    def create_train_df(self, paragrpahs: UnstructuredParagraph) -> pd.DataFrame:
        raise NotImplementedError()

class PptxParse(UnstructuredParse):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        try:
            self.PptxLoader = UnstructuredPowerPointLoader(file_path=self._filepath, mode = "elements", 
                                                       post_processors=self._post_processors)
        except Exception as e:
            print(e.__str__())
            print("Cannot process file:" , filepath)
    
    def process_elements(self) -> Tuple[Union[UnstructuredParagraph, str], bool]:
        rows = []
        try:
            docs = self.PptxLoader.load()
            for doc in docs:
                text = doc.page_content
                text = str(text).strip().replace("\r\n", " ").replace("\n", " ").replace("\t", " ").replace(",", " ")
                sents = sent_tokenize(text)
                sents = [
                    sent.replace("\t", " ").replace(",", " ").replace("\n", " ").strip().lower()
                    for sent in sents
                ]
                para = " ".join(sents)
                row = UnstructuredParagraph(para=para, filepath=doc.metadata['filename'], page_no=doc.metadata['page_number'], display=str(text.replace("\n", " ")))
                rows.append(row)
                    
            return rows, True
        except Exception as e:
            print(e.__str__())
            return None, False
        
    def create_train_df(self, paragrpahs:List[UnstructuredParagraph]) -> pd.DataFrame:
        df = pd.DataFrame(index=range(len(paragrpahs)), columns=["para", "filename", "page", "display"])
        for i, elem in enumerate(paragrpahs):
            
            df.iloc[i] = [
                elem.para,
                elem.filepath,
                elem.page_no,
                elem.display
            ]
            
        for column in ["para", "display"]:
            df[column] = df[column].apply(utils.ensure_valid_encoding)
        return df
        


class EmlParse(UnstructuredParse):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        try:
            self.EmlLoader = UnstructuredEmailLoader(file_path=self._filepath, mode = "elements", 
                                                       post_processors=self._post_processors)
        except Exception as e:
            print(e.__str__())
            print("Cannot process file:" , filepath)
    
    def process_elements(self) -> Tuple[Union[EmlParagraph, str], bool]:
        rows = []
        try:
            docs = self.EmlLoader.load()
            for doc in docs:
                text = doc.page_content
                text = str(text).strip().replace("\r\n", " ").replace("\n", " ").replace("\t", " ").replace(",", " ")
                sents = sent_tokenize(text)
                sents = [
                    sent.replace("\t", " ").replace(",", " ").replace("\n", " ").strip().lower()
                    for sent in sents
                ]
                para = " ".join(sents)
                row = EmlParagraph(para=para, filepath=doc.metadata['filename'], 
                                            page_no=doc.metadata['page_number'], display=str(text.replace("\n", " "))
                                            subject=doc.metadata['subject'], sent_from=doc.metadata['sent_from'], sent_to=doc.metadata['sent_to'])
                rows.append(row)
                    
            return rows, True
        except Exception as e:
            print(e.__str__())
            return "Cannot process Eml file: " + self._filepath, False
        
    def create_train_df(self, paragrpahs:List[EmlParagraph]) -> pd.DataFrame:
        df = pd.DataFrame(index=range(len(paragrpahs)), columns=["para", "filename", "page", "display", "subject", "sent_from", "sent_to"])
        for i, elem in enumerate(paragrpahs):
            
            df.iloc[i] = [
                elem.para,
                elem.filepath,
                elem.page_no,
                elem.display,
                elem.subject,
                elem.sent_from,
                elem.sent_to,
            ]
            
        for column in ["para", "display", "subject", "sent_from", "sent_to"]:
            df[column] = df[column].apply(utils.ensure_valid_encoding)
        return df
        

class TxtParse(UnstructuredParse):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        try:
            self.TxtLoader = UnstructuredFileLoader(file_path=self._filepath, mode = "single", post_processors=self._post_processors)
        except Exception as e:
            print(str(e))
            print("Cannot process file: " , filepath)


    def get_elements(self) -> Tuple[Union[UnstructuredParagraph, str], bool]:
        try:
            doc = self.TxtLoader.load()
            content = str(doc[0].page_content).strip().replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
            chunks = chunk_text(content)

            paragraphs = [UnstructuredParagraph(para = chunk, filepath = self._filepath, page_no = None, display = chunk) for chunk in chunks]
            return paragraphs
        except Exception as e:
            print(str(e))
            return "Cannot process Text file: " + self._filepath, False
        
    def create_train_df(self, paragraphs: List[UnstructuredParagraph]) -> pd.DataFrame:
        df = pd.DataFrame(index = range(len(paragraphs)), columns = ["para", "filepath", "page", "display"])

        for idx, paragraph in enumerate(paragraphs):
            sentences = sent_tokenize(paragraph.para)
            sentences = [
                sent.replace("\t", " ").replace(",", " ").replace("\n", " ").strip().lower()
                for sent in sentences
            ]

            para = " ".join(sentences)
            df.iloc[idx] = [
                para,
                paragraph.filepath,
                paragraph.page_no,
                paragraph.display
            ]
        for column in ["para", "display"]:
            df[column] = df[column].apply(utils.ensure_valid_encoding)
        
        return df, True 

class DocParse(UnstructuredParse):
    pass


    
    
