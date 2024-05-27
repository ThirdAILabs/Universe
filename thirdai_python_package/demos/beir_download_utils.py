import csv
import json
import logging
import os
import zipfile
from typing import Dict, Tuple

import requests
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


def write_unsupervised_file(corpus, data_path):
    unsup_file = data_path + "/unsupervised.csv"
    with open(unsup_file, "w") as fw:
        header = "DOC_ID,TITLE,TEXT\n"
        fw.write(header)
        count = 0
        for key in corpus:
            title = corpus[key]["title"].replace(",", " ")
            title = title.replace("\r", " ")
            title = title.replace("\n", " ")
            title = " " if title == "" else title
            text = corpus[key]["text"].replace(",", " ")
            text = text.replace("\r", " ")
            text = text.replace("\n", " ")
            text = " " if text == "" else text
            fw.write(str(count) + "," + title + "," + text + "\n")
            count += 1


def remap_doc_ids(corpus):
    doc_ids_to_integers = {}
    count = 0
    for key in corpus:
        doc_ids_to_integers[key] = count
        count += 1
    return doc_ids_to_integers


def remap_query_answers(qrels, doc_ids_to_integers):
    new_qrels = {}
    for key in qrels:
        output = {}
        for doc_id in qrels[key]:
            if qrels[key][doc_id] > 0:
                output[str(doc_ids_to_integers[doc_id])] = qrels[key][doc_id]
        new_qrels[key] = output
    return new_qrels


def write_supervised_file(queries, answers, data_path, filename):
    sup_train_file = data_path + "/" + filename
    with open(sup_train_file, "w") as fw:
        fw.write("QUERY,DOC_ID\n")

        for key in queries:
            query = queries[key].replace(",", " ")
            doc_ids = ":".join(list(answers[key].keys()))
            fw.write(query + "," + doc_ids + "\n")


#############################################################################
# Everything below this was taken from https://github.com/beir-cellar/beir
# They have an Apache 2.0 license so it should be good for commercial and private
# use. We include their logic here so as to not add the additional dependency
#############################################################################


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with open(save_path, "wb") as fd, tqdm(
        desc=save_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()


def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        print(f"Downloading {dataset} ...")
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        print(f"Unzipping {dataset} ...")
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))


class GenericDataLoader:
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = (
            os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        )
        self.query_file = (
            os.path.join(data_folder, query_file) if data_folder else query_file
        )
        self.qrels_folder = (
            os.path.join(data_folder, qrels_folder) if data_folder else None
        )
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(
                "File {} not present! Please provide accurate file.".format(fIn)
            )

        if not fIn.endswith(ext):
            raise ValueError(
                "File {} must be present with extension {}".format(fIn, ext)
            )

    def load_custom(
        self,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(
        self, split="test"
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):
        reader = csv.reader(
            open(self.qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )

        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
