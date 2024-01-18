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
            unsup_prefix = "unsup" if config.trn_supervised_path else ""
            cls.log_precision_metrics(db, config, prefix=unsup_prefix)
            mlflow.log_metric("insertion time", time.time() - start_insert_time)

        if config.trn_supervised_path:
            trn_supervised = os.path.join(path_prefix, config.trn_supervised_path)
            start_sup_train = time.time()
            sup_doc = ndb.Sup(
                trn_supervised,
                query_column=config.query_column,
                id_column=config.id_column,
                id_delimiter=config.id_delimiter,
                source_id=source_ids[0],
            )
            db.supervised_train([sup_doc], learning_rate=0.001, epochs=10)

            if mlflow_logger:
                cls.log_precision_metrics(db, config, prefix="sup")
                mlflow.log_metric("sup training time", time.time() - start_sup_train)

        if mlflow_logger:
            test_df = pd.read_csv(config.tst_sets[0])
            num_queries = 20
            total_query_time = 0
            for i in range(num_queries):
                start_query_time = time.perf_counter_ns()
                db.search(test_df[config.query_column][i % len(test_df)], top_k=5)
                total_query_time += time.perf_counter_ns() - start_query_time
            mlflow.log_metric("query time", total_query_time / num_queries)

    @classmethod
    def log_precision_metrics(cls, db, config, prefix):
        tst_sets = config.tst_sets
        if len(tst_sets) == 0:
            precision = cls.get_precision(
                db, tst_sets[0], config.query_col, config.doc_col, config.id_delimiter
            )
            mlflow.log_metric(f"{prefix} P at 1", precision)
            return

        for tst_set in tst_sets:
            precision = cls.get_precision(
                db, tst_set, config.query_col, config.doc_col, config.id_delimiter
            )
            tst_set_name = tst_set.split("/")[-1].split(".")[0]
            mlflow.log_metric(f"{prefix} P at 1 {tst_set_name}", precision)

    @classmethod
    def get_precision(cls, db, tst_set, query_col, doc_col, id_delimiter):
        test_df = pd.read_csv(tst_set)
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
