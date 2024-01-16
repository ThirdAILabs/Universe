from abc import ABC


class NDBConfig(ABC):
    config_name = None
    dataset_name = None
    unsupervised_path = f"{dataset_name}/unsupervised.csv"

    id_column = "DOC_ID"
    strong_columns = ["TITLE"]
    weak_columns = ["TEXT"]
    query_column = "QUERY"
    id_delimiter = ":"

    trn_supervised_path = f"{dataset_name}/trn_supervised.csv"
    tst_supervised_path = f"{dataset_name}/tst_supervised.csv"


class ScifactNDBConfig(NDBConfig):
    config_name = "scifact_ndb"
    dataset_name = "scifact"


class FiqaNDBConfig(NDBConfig):
    config_name = "fiqa_ndb"
    dataset_name = "fiqa"

    trn_supervised_path = None


class TrecCovidNDBConfig(NDBConfig):
    config_name = "trec_covid_ndb"
    dataset_name = "trec-covid"

    trn_supervised_path = None


class CookingNDBConfig(NDBConfig):
    config_name = "cooking_ndb"
    dataset_name = "cooking"

    id_column = "LABEL_IDS"
    strong_columns = ["TITLE", "BULLET_POINTS"]
    weak_column = ["DESCRIPTION", "BRAND"]
    id_delimiter = ";"


class WayfairNDBConfig(NDBConfig):
    config_name = "wayfair_ndb"
    dataset_name = "wayfair"

    id_column = "LABEL_IDS"
    strong_columns = ["TITLE", "BULLET_POINTS"]
    weak_column = ["DESCRIPTION", "BRAND"]
    id_delimiter = ";"


class Amazon1_3MConfig(NDBConfig):
    config_name = "amazontitles-1.3mm_ndb"
    dataset_name = "amazontitles-1.3mm"

    strong_columns = []
    weak_columns = ["TITLE"]


class Pubmed800kConfig(NDBConfig):
    config_name = "pubmed_800k_ndb"
    dataset_name = "pubmed_800k"
