# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
from langchain.document_loaders import UnstructuredFileLoader
from nltk.tokenize import sent_tokenize
import parsing_utils.utils as utils

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


def process_unstructured_file(filename):
    try:
        rows = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 75,
            length_function = len,
        )
        loader = UnstructuredFileLoader(filename)
        docs = loader.load()
        content = docs[0].page_content
        texts = text_splitter.split_text(content)
        for text in texts:
            text = str(text).strip().replace("\r\n", " ").replace("\n", " ").replace("\t", " ").replace(",", " ")
            row = [text, filename]
            rows.append(row)

        # paras = [
        #     (chunk, page_no, docname, temp) 
        #     for passage, page_no, docname, temp in paras
        #     for chunk in utils.chunk_text(passage)
        # ]
        # for para in paras:
        #     if len(para) > 0:
        #         sent = re.sub(
        #             " +",
        #             " ",
        #             str(para[0])
        #             .replace("\t", " ")
        #             .replace(",", " ")
        #             .replace("\n", " ")
        #             .strip(),
        #         )
        #         if len(sent) > 0:
        #             rows.append((count, sent.lower(), sent, para[1], para[2], str(para[3])))
        #             count += 1
                
        return rows, True
    except Exception as e:
        print(e.__str__())
        return "Cannot process pdf file:" + filename, False
    
def create_train_df(elements, id_col):
    count = 0
    df = pd.DataFrame(index=range(len(elements)), columns=[id_col, "passage", "para", "filename", "page", "display"])
    for i, elem in enumerate(elements):
        sents = sent_tokenize(elem[0])
        sents = list(map(lambda x: x.lower(), sents))
        passage = " ".join(sents)
        df.iloc[i] = [elem[-1], passage, passage, elem[1], 0, passage]
        count = count + 1
    for column in ["passage"]:
        df[column] = df[column].apply(utils.ensure_valid_encoding)
    return df



if __name__ == "__main__":
    process_unstructured_file("1706.03762.pdf")
    
