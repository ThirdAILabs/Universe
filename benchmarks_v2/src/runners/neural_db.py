import os
import shutil

import mlflow
import pandas as pd
import thirdai.neural_db as ndb

from ..configs.neural_db_configs import NDBConfig
from ..runners.runner import Runner


class NDBRunner(Runner):
    config_type = NDBConfig

    @classmethod
    def run_benchmark(cls, config: NDBConfig, path_prefix: str, mlflow_logger):
        db = ndb.NeuralDB()
        doc = ndb.CSV(
            path=os.path.join(path_prefix, config.unsupervised_path),
            id_column=config.id_column,
            strong_columns=config.strong_columns,
            weak_columns=config.weak_columns,
        )

        source_ids = db.insert(sources=[doc], train=True)

        tst_supervised = os.path.join(path_prefix, config.tst_supervised_path)
        precision = cls.get_precision(db, tst_supervised)

        if mlflow_logger:
            mlflow.log_metric("cs_p_at_1", precision)

        trn_supervised = os.path.join(path_prefix, config.trn_supervised_path)
        if os.path.exists():
            sup_doc = ndb.Sup(
                trn_supervised,
                query_column=config.query_column,
                id_column=config.id_column,
                id_delimiter=config.id_delimiter,
                source_id=source_ids[0],
            )
            db.supervised_train([sup_doc], learning_rate=0.001, epochs=10)

            precision = cls.get_precision(db, tst_supervised)

            if mlflow_logger:
                mlflow.log_metric("sup_p_at_1", precision)

    @classmethod
    def get_precision(db, tst_supervised, query_col, doc_col, id_delimiter):
        test_df = pd.read_csv(tst_supervised)
        correct_count = 0
        batched_results = db.search_batch(queries=list(test_df[query_col]), top_k=1)
        for i in range(test_df.shape[0]):
            top_pred = batched_results[i][0].id
            if not id_delimiter:
                relevant_ids = [str(test_df[doc_col][i])]
            else:
                relevant_ids = [x for x in test_df[doc_col][i].split(id_delimiter)]
            if str(top_pred) in relevant_ids:
                correct_count += 1
        return correct_count / test_df.shape[0]
