from abc import ABC

from thirdai import data


class NDBConfig(ABC):
    config_name = None
    config_type = "ndb_small"
    dataset_name = None
    unsupervised_path = None

    id_column = "DOC_ID"
    strong_columns = ["TITLE"]
    weak_columns = ["TEXT"]
    query_column = "QUERY"
    id_delimiter = ":"

    trn_supervised_path = None
    tst_sets = []

    vlc_config = data.transformations.VariableLengthConfig()


class ScifactNDBConfig(NDBConfig):
    config_name = "scifact_ndb"
    dataset_name = "scifact"

    unsupervised_path = "scifact/unsupervised.csv"
    trn_supervised_path = "scifact/trn_supervised.csv"
    tst_sets = ["scifact/tst_supervised.csv"]


class FiqaNDBConfig(NDBConfig):
    config_name = "fiqa_ndb"
    dataset_name = "fiqa"

    unsupervised_path = "fiqa/unsupervised.csv"
    tst_sets = ["fiqa/tst_supervised.csv"]


class TrecCovidNDBConfig(NDBConfig):
    config_name = "trec_covid_ndb"
    dataset_name = "trec-covid"

    unsupervised_path = "trec-covid/unsupervised.csv"
    tst_sets = ["trec-covid/tst_supervised.csv"]


class CookingNDBConfig(NDBConfig):
    config_name = "cooking_ndb"
    dataset_name = "cooking"

    id_column = "LABEL_IDS"
    strong_columns = ["TITLE", "BULLET_POINTS"]
    weak_columns = ["DESCRIPTION", "BRAND"]
    id_delimiter = ";"

    unsupervised_path = "catalog_recommender/cooking/reformatted_trn_unsupervised.csv"
    tst_sets = ["catalog_recommender/cooking/reformatted_tst_supervised.csv"]


class Wiki5KNDBConfig(NDBConfig):
    config_name = "wiki_5k_ndb"
    dataset_name = "wiki-5k"

    strong_columns = []

    unsupervised_path = "wiki/unsupervised_small.csv"
    tst_sets = [
        f"wiki/len_{l}{perturbation}.csv"
        for l in [5, 10, 15, 20]
        for perturbation in ["", "_perturbed"]
    ]


class Wiki105KNDBConfig(NDBConfig):
    config_name = "wiki_105k_ndb"
    dataset_name = "wiki-105k"

    strong_columns = []

    unsupervised_path = "wiki/unsupervised_large.csv"
    tst_sets = [
        f"wiki/len_{l}{perturbation}.csv"
        for l in [5, 10, 15, 20]
        for perturbation in ["", "_perturbed"]
    ]


class Amazon1_3MConfig(NDBConfig):
    config_name = "amazontitles-1.3mm_ndb"
    config_type = "ndb_large"
    dataset_name = "amazontitles-1.3mm"

    unsupervised_path = "amazontitles-1.3mm/unsupervised.csv"
    trn_supervised_path = "amazontitles-1.3mm/trn_supervised.csv"
    tst_sets = ["amazontitles-1.3mm/tst_supervised.csv"]

    strong_columns = []
    weak_columns = ["TITLE"]

    vlc_config = data.transformations.VariableLengthConfig(
        num_slices=3,
    )


class Pubmed800kConfig(NDBConfig):
    config_name = "pubmed_800k_ndb"
    config_type = "ndb_large"
    dataset_name = "pubmed-800k"

    id_column = "label"
    strong_columns = ["title"]
    weak_columns = ["abstract"]
    query_column = "query"

    unsupervised_path = "pubmed_800k/unsupervised.csv"
    tst_sets = ["pubmed_800k/titles.csv", "pubmed_800k/abstracts.csv"]

    vlc_config = data.transformations.VariableLengthConfig(
        num_slices=13,
        strong_sample_num_words=7,
    )
