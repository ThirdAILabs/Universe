import os
import time

import mlflow
import pandas as pd
import thirdai.neural_db as ndb

from ..configs.ndb_configs import NDBConfig
from .runner import Runner


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

        start_insert_time = time.time()
        source_ids = db.insert(sources=[doc], train=True)
        if mlflow_logger:
            mlflow.log_metric("insertion time", time.time() - start_insert_time)

        tst_supervised = os.path.join(path_prefix, config.tst_supervised_path)
        precision = cls.get_precision(
            db,
            tst_supervised,
            config.query_column,
            config.doc_col,
            config.id_delimiter,
        )

        if mlflow_logger:
            mlflow.log_metric("unsup precision at 1", precision)

        trn_supervised = os.path.join(path_prefix, config.trn_supervised_path)
        if os.path.exists():
            start_sup_train_time = time.time()
            sup_doc = ndb.Sup(
                trn_supervised,
                query_column=config.query_column,
                id_column=config.id_column,
                id_delimiter=config.id_delimiter,
                source_id=source_ids[0],
            )
            db.supervised_train([sup_doc], learning_rate=0.001, epochs=10)

            precision = cls.get_precision(
                db,
                tst_supervised,
                config.query_column,
                config.doc_col,
                config.id_delimiter,
            )

            if mlflow_logger:
                mlflow.log_metric("sup precision at 1", precision)
                mlflow.log_metric(
                    "sup training time", time.time() - start_sup_train_time
                )

        if mlflow_logger:
            test_df = pd.read_csv(tst_supervised)
            num_queries = 20
            total_query_time = 0
            for i in range(num_queries):
                start_query_time = time.perf_counter_ns()
                db.search(test_df[config.query_column][i % len(test_df)], top_k=5)
                total_query_time += time.perf_counter_ns() - start_query_time
            mlflow.log_metric("avg query time", total_query_time / num_queries)

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
