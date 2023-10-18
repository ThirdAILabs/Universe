import copy
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import thirdai
import unidecode
from thirdai._thirdai import bolt
from thirdai.dataset.data_source import PyDataSource

from . import loggers, teachers
from .documents import CSV, Document, DocumentManager, Reference
from .models import CancelState, Mach
from .savable_state import State

Strength = Enum("Strength", ["Weak", "Medium", "Strong"])


def no_op(*args, **kwargs):
    pass


"""Sup and SupDataSource are classes that manage entity IDs for supervised
training.

Entity = an item that can be retrieved by NeuralDB. If we insert an ndb.CSV
object into NeuralDB, then each row of the CSV file is an entity. If we insert 
an ndb.PDF object, then each paragraph is an entity. If we insert an 
ndb.SentenceLevelDOCX object, then each sentence is an entity.

If this still doesn't make sense, consider a scenario where you insert a CSV 
file into NeuralDB and want to improve the performance of the database by
training it on supervised training samples. That is, you want the model to 
learn from (query, ID) pairs.

Since you only inserted one file, the ID of each entity in NeuralDB's index
is the same as the ID given in the file. Thus, the model can directly ingest
the (query, ID) pairs from your supervised dataset. However, this is not the 
case if you inserted multiple CSV files. For example, suppose you insert file A 
containing entities with IDs 0 through 100 and also insert file B containing 
its own set of entities with IDs 0 through 100. To disambiguate between entities
from different files, NeuralDB automatically offsets the IDs of the second file.
Consequently, you also have to adjust the labels of supervised training samples
corresponding to entities in file B. 

Instead of leaking the abstraction by making the user manually change the labels
of their dataset, we created Sup and SupDataSource to handle this.

If the user would rather use the database-assigned IDs instead of IDs from the 
original document, this can be done by passing uses_db_id = True to Sup(). This
is useful for cases where the user does not know the IDs of the entities in the
original documents. For example, if the original document is a PDF, then it is
NeuralDB that parses it into paragraphs; the user does not know the ID of each
paragraph beforehand. In this scenario, it is much easier for the user to just
use the database-assigned IDs.
"""


class Sup:
    """An object that contains supervised samples. This object is to be passed
    into NeuralDB.supervised_train().

    It can be initialized either with a CSV file, in which case it needs query
    and ID column names, or with sequences of queries and labels. It also needs
    to know which source object (i.e. which inserted CSV or PDF object) contains
    the relevant entities to the supervised samples.

    If uses_db_id is True, then the labels are assumed to use database-assigned
    IDs and will not be converted.
    """

    def __init__(
        self,
        csv: str = None,
        query_column: str = None,
        id_column: str = None,
        id_delimiter: str = None,
        queries: Sequence[str] = None,
        labels: Sequence[Sequence[int]] = None,
        source_id: str = "",
        uses_db_id: bool = False,
    ):
        if csv is not None and query_column is not None and id_column is not None:
            df = pd.read_csv(csv)
            self.queries = df[query_column]
            self.labels = df[id_column]
            for i, label in enumerate(self.labels):
                if label == None or label == "":
                    raise ValueError(
                        f"Got a supervised sample with an empty label, query: '{self.queries[i]}'"
                    )
            if id_delimiter:
                self.labels = self.labels.apply(
                    lambda label: list(
                        map(int, str(label).strip(id_delimiter).split(id_delimiter))
                    )
                )
            else:
                self.labels = self.labels.apply(lambda label: [int(label)])

        elif queries is not None and labels is not None:
            if len(queries) != len(labels):
                raise ValueError(
                    "Queries and labels sequences must be the same length."
                )
            self.queries = queries
            self.labels = labels
        # elif csv is None and
        else:
            raise ValueError(
                "Sup must be initialized with csv, query_column and id_column, or queries and labels."
            )
        self.source_id = source_id
        self.uses_db_id = uses_db_id


class SupDataSource(PyDataSource):
    """Combines supervised samples from multiple Sup objects into a single data
    source. This allows NeuralDB's underlying model to train on all provided
    supervised datasets simultaneously.
    """

    def __init__(
        self,
        doc_manager: DocumentManager,
        query_col: str,
        data: List[Sup],
        id_delimiter: Optional[str],
    ):
        PyDataSource.__init__(self)
        self.doc_manager = doc_manager
        self.query_col = query_col
        self.data = data
        self.id_delimiter = id_delimiter
        if not self.id_delimiter:
            print("WARNING: this model does not fully support multi-label datasets.")
        self.restart()

    def _csv_line(self, query: str, label: str):
        df = pd.DataFrame(
            {
                self.query_col: [query],
                self.doc_manager.id_column: [label],
            }
        )
        return df.to_csv(header=None, index=None).strip("\n")

    def _id_offset(self, sup: Sup):
        if sup.uses_db_id:
            return 0
        source_ids = self.doc_manager.match_source_id_by_prefix(sup.source_id)
        if len(source_ids) == 0:
            raise ValueError(f"Cannot find source with id {sup.source_id}")
        if len(source_ids) > 1:
            raise ValueError(f"Multiple sources match the prefix {sup.source_id}")
        _, start_id = self.doc_manager.source_by_id(source_ids[0])
        return start_id

    def _get_line_iterator(self):
        # First yield the header
        yield self._csv_line(self.query_col, self.doc_manager.id_column)
        # Then yield rows
        for sup in self.data:
            id_offset = self._id_offset(sup)
            for query, labels in zip(sup.queries, sup.labels):
                if self.id_delimiter:
                    label_str = self.id_delimiter.join(
                        [str(id_offset + label) for label in labels]
                    )
                    yield self._csv_line(query, label_str)
                else:
                    for label in labels:
                        label_str = str(id_offset + label)
                        yield self._csv_line(query, label_str)

    def resource_name(self) -> str:
        return "Supervised training samples"


class NeuralDB:
    def __init__(self, user_id: str = "user", **kwargs) -> None:
        """user_id is used for logging purposes only"""
        self._user_id: str = user_id

        # The savable_state kwarg is only used in static constructor methods
        # and should not be used by an external user.
        # We read savable_state from kwargs so that it doesn't appear in the
        # arguments list and confuse users.
        if "savable_state" not in kwargs:
            self._savable_state: State = State(
                model=Mach(id_col="id", query_col="query", **kwargs),
                logger=loggers.LoggerList([loggers.InMemoryLogger()]),
            )
        else:
            self._savable_state = kwargs["savable_state"]

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        user_id: str = "user",
        on_progress: Callable = no_op,
    ):
        checkpoint_path = Path(checkpoint_path)
        savable_state = State.load(checkpoint_path, on_progress)
        if savable_state.model and savable_state.model.get_model():
            savable_state.model.get_model().set_mach_sampling_threshold(0.01)
        if not isinstance(savable_state.logger, loggers.LoggerList):
            # TODO(Geordie / Yash): Add DBLogger to LoggerList once ready.
            savable_state.logger = loggers.LoggerList([savable_state.logger])

        return NeuralDB(user_id, savable_state=savable_state)

    @staticmethod
    def from_udt(
        udt: bolt.UniversalDeepTransformer,
        user_id: str = "user",
        csv: Optional[str] = None,
        csv_id_column: Optional[str] = None,
        csv_strong_columns: Optional[List[str]] = None,
        csv_weak_columns: Optional[List[str]] = None,
        csv_reference_columns: Optional[List[str]] = None,
    ):
        """Instantiate a NeuralDB, using the given UDT as the underlying model.
        Usually for porting a pretrained model into the NeuralDB format.
        Use the optional csv-related arguments to insert the pretraining dataset
        into the NeuralDB instance.
        """
        if csv is None:
            udt.clear_index()

        udt.enable_rlhf()
        udt.set_mach_sampling_threshold(0.01)
        fhr, emb_dim, out_dim = udt.model_dims()
        data_types = udt.data_types()

        if len(data_types) != 2:
            raise ValueError(
                f"Incompatible UDT model. Expected two data types but found {len(data_types)}."
            )
        query_col = None
        id_col = None
        id_delimiter = None
        for column, dtype in data_types.items():
            if isinstance(dtype, bolt.types.text):
                query_col = column
            if isinstance(dtype, bolt.types.categorical):
                id_col = column
                id_delimiter = dtype.delimiter
        if query_col is None:
            raise ValueError(f"Incompatible UDT model. Cannot find a query column.")
        if id_col is None:
            raise ValueError(f"Incompatible UDT model. Cannot find an id column.")

        model = Mach(
            id_col=id_col,
            id_delimiter=id_delimiter,
            query_col=query_col,
            fhr=fhr,
            embedding_dimension=emb_dim,
            extreme_output_dim=out_dim,
        )
        model.model = udt
        logger = loggers.LoggerList([loggers.InMemoryLogger()])
        savable_state = State(model=model, logger=logger)

        if csv is not None:
            if (
                csv_id_column is None
                or csv_strong_columns is None
                or csv_weak_columns is None
                or csv_reference_columns is None
            ):
                error_msg = "If the `csv` arg is provided, then the following args must also be provided:\n"
                error_msg += " - `csv_id_column`\n"
                error_msg += " - `csv_strong_columns`\n"
                error_msg += " - `csv_weak_columns`\n"
                error_msg += " - `csv_reference_columns`\n"
                raise ValueError(error_msg)
            csv_doc = CSV(
                path=csv,
                id_column=csv_id_column,
                strong_columns=csv_strong_columns,
                weak_columns=csv_weak_columns,
                reference_columns=csv_reference_columns,
            )
            savable_state.documents.add([csv_doc])
            savable_state.model.set_n_ids(csv_doc.size)

        return NeuralDB(user_id, savable_state=savable_state)

    def pretrain_distributed(
        self,
        documents,
        scaling_config,
        run_config,
        learning_rate: float = 0.001,
        epochs: int = 5,
        batch_size: int = None,
        metrics: List[str] = [],
        max_in_memory_batches: Optional[int] = None,
        communication_backend="gloo",
        log_folder=None,
    ):
        """
        Pretrains a model in a distributed manner using the provided documents.

        Args:
            documents: List of documents for pretraining. All the documents must have the same id column.
            scaling_config: Configuration related to the scaling aspects for Ray trainer. Read
                https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
            run_config: Configuration related to the runtime aspects for Ray trainer. Read
                https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html
                ** Note: We need to specify `storage_path` in `RunConfig` which must be a networked **
                ** file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards) **
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
            epochs (int, optional): Number of epochs to train. Default is 5.
            batch_size (int, optional): Size of each batch for training. If not provided, will be determined automatically.
            metrics (List[str], optional): List of metrics to evaluate during training. Default is an empty list.
            max_in_memory_batches (Optional[int], optional): Number of batches to load in memory at once. Useful for
                streaming support when dataset is too large to fit in memory. If None, all batches will be loaded.
            communication_backend (str, optional): Bolt Distributed Training uses Torch Communication Backend. This
                refers to backend for inter-worker communication. Default is "gloo".

        Notes:
            - Make sure to pass id_column to neural_db.CSV() making sure the ids are in ascending order starting from 0.
            - The `scaling_config`, `run_config`, and `resume_from_checkpoint` arguments are related to the Ray trainer configuration. Read
                https://docs.ray.io/en/latest/ray-air/trainers.html#trainer-basics
            - Ensure that the communication backend specified is compatible with the hardware and network setup for MPI/Gloo backend.
        """

        import warnings
        from distutils.version import LooseVersion

        import ray
        import thirdai.distributed_bolt as dist
        from ray.train.torch import TorchConfig

        ray_version = ray.__version__
        if LooseVersion(ray_version) >= LooseVersion("2.7"):
            warnings.warn(
                """
                Using ray version 2.7 or higher requires specifying a remote or NFS storage path. 
                Support for local checkpoints has been discontinued in these versions. 
                Refer to https://github.com/ray-project/ray/issues/37177 for details.
                """.strip()
            )

        if not isinstance(documents, list) or not all(
            isinstance(doc, CSV) for doc in documents
        ):
            raise ValueError(
                "The pretrain_distributed function currently only supports CSV documents."
            )

        def training_loop_per_worker(config):
            import os

            import thirdai.distributed_bolt as dist
            from ray import train
            from thirdai.dataset import RayCsvDataSource

            if config["licensing_lambda"]:
                config["licensing_lambda"]()

            strong_column_names = config["strong_column_names"]
            weak_column_names = config["weak_column_names"]
            learning_rate = config["learning_rate"]
            epochs = config["epochs"]
            batch_size = config["batch_size"]
            metrics = config["metrics"]
            max_in_memory_batches = config["max_in_memory_batches"]
            model_ref = config["model_ref"]
            model_target_column = config["model_target_col"]
            document_target_col = config["document_target_col"]
            log_folder = train_loop_config["log_folder"]

            # ray data will automatically split the data if the dataset is passed with key "train"
            # to training loop. Read https://docs.ray.io/en/latest/ray-air/check-ingest.html#splitting-data-across-workers
            stream_split_data_iterator = train.get_dataset_shard("train")

            model = ray.get(model_ref)

            if log_folder:
                if not os.path.exists(log_folder):
                    print(f"Folder '{log_folder}' does not exist. Creating it...")
                    os.makedirs(log_folder)
                    print(f"Folder '{log_folder}' created successfully!")
                thirdai.logging.setup(
                    log_to_stderr=False,
                    path=os.path.join(
                        log_folder, f"worker-{train.get_context().get_world_rank()}.log"
                    ),
                    level="info",
                )

            metrics = model.coldstart_distributed_on_data_source(
                data_source=RayCsvDataSource(
                    stream_split_data_iterator, model_target_column, document_target_col
                ),
                strong_column_names=strong_column_names,
                weak_column_names=weak_column_names,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                metrics=metrics,
                max_in_memory_batches=max_in_memory_batches,
            )

            rank = train.get_context().get_world_rank()
            checkpoint = None
            if rank == 0:
                # Use `with_optimizers=False` to save model without optimizer states
                checkpoint = dist.UDTCheckPoint.from_model(model, with_optimizers=False)

            train.report(metrics=metrics, checkpoint=checkpoint)

        csv_paths = [str(document.path.resolve()) for document in documents]

        train_ray_ds = ray.data.read_csv(csv_paths)

        train_loop_config = {}

        # we cannot pass the model directly to config given config results in OOM very frequently with bigger model.
        model_ref = ray.put(self._savable_state.model.get_model())

        # If this is a file based license, it will assume the license to available at the same location on each of the
        # machine
        licensing_lambda = None
        if hasattr(thirdai._thirdai, "licensing"):
            license_state = thirdai._thirdai.licensing._get_license_state()
            licensing_lambda = lambda: thirdai._thirdai.licensing._set_license_state(
                license_state
            )

        train_loop_config["licensing_lambda"] = licensing_lambda
        train_loop_config["strong_column_names"] = documents[0].strong_columns
        train_loop_config["weak_column_names"] = documents[0].weak_columns
        train_loop_config["learning_rate"] = learning_rate
        train_loop_config["epochs"] = epochs
        train_loop_config["batch_size"] = batch_size
        train_loop_config["metrics"] = metrics
        train_loop_config["max_in_memory_batches"] = max_in_memory_batches
        train_loop_config["model_ref"] = model_ref
        train_loop_config["model_target_col"] = self._savable_state.model.get_id_col()
        # Note(pratik): We are having an assumption here, that each of the document must have the
        # same target column
        train_loop_config["document_target_col"] = documents[0].id_column
        train_loop_config["log_folder"] = log_folder

        trainer = dist.BoltTrainer(
            train_loop_per_worker=training_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            backend_config=TorchConfig(backend=communication_backend),
            datasets={"train": train_ray_ds},
            run_config=run_config,
        )

        result_and_checkpoint = trainer.fit()

        # TODO(pratik/mritunjay): This will stop working with ray==2.7 if runconfig doesnt specify s3 storage path.
        # Update: https://github.com/ThirdAILabs/Universe/pull/1784
        # `run_config` is made required argument in `pretrained_distributed` function
        model = dist.UDTCheckPoint.get_model(result_and_checkpoint.checkpoint)

        self._savable_state.model.set_model(model)

    def ready_to_search(self) -> bool:
        """Returns True if documents have been inserted and the model is
        prepared to serve queries, False otherwise.
        """
        return self._savable_state.ready()

    def sources(self) -> Dict[str, Document]:
        """Returns a mapping from source IDs to their corresponding document
        objects. This is useful when you need to know the source ID of a
        document you inserted, e.g. for creating a Sup object for
        supervised_train().
        """
        return self._savable_state.documents.sources()

    def save(self, save_to: str, on_progress: Callable = no_op) -> str:
        return self._savable_state.save(Path(save_to), on_progress)

    def insert(
        self,
        sources: List[Document],
        train: bool = True,
        fast_approximation: bool = True,
        num_buckets_to_sample: Optional[int] = None,
        on_progress: Callable = no_op,
        on_success: Callable = no_op,
        on_error: Callable = None,
        cancel_state: CancelState = None,
    ) -> List[str]:
        """Inserts sources into the database.
        fast_approximation: much faster insertion with a slight drop in
        performance.
        num_buckets_to_sample: when assigning set of MACH buckets to an entity,
        look at the top num_buckets_to_sample buckets, then choose the least
        occupied ones. This prevents MACH buckets from overcrowding,
        cancel_state: an object that can be used to stop an ongoing insertion.
        Primarily used for PocketLLM.
        """
        documents_copy = copy.deepcopy(self._savable_state.documents)
        try:
            intro_and_train, ids = self._savable_state.documents.add(sources)
        except Exception as e:
            self._savable_state.documents = documents_copy
            if on_error is not None:
                on_error(error_msg=f"Failed to add files. {e.__str__()}")
                return []
            raise e

        self._savable_state.model.index_documents(
            intro_documents=intro_and_train.intro,
            train_documents=intro_and_train.train,
            num_buckets_to_sample=num_buckets_to_sample,
            fast_approximation=fast_approximation,
            should_train=train,
            on_progress=on_progress,
            cancel_state=cancel_state,
        )

        self._savable_state.logger.log(
            session_id=self._user_id,
            action="Train",
            args={"files": intro_and_train.intro.resource_name()},
        )

        on_success()
        return ids

    def clear_sources(self) -> None:
        self._savable_state.documents.clear()
        self._savable_state.model.forget_documents()

    def search(
        self, query: str, top_k: int, constraints=None, on_error: Callable = None
    ) -> List[Reference]:
        matching_entities = None
        if constraints:
            matching_entities = self._savable_state.documents.entity_ids_by_constraints(
                constraints
            )
            result_ids = self._savable_state.model.score(
                samples=[query], entities=[matching_entities], n_results=top_k
            )[0]
        else:
            result_ids = self._savable_state.model.infer_labels(
                samples=[query], n_results=top_k
            )[0]

        references = []
        for rid, score in result_ids:
            ref = self._savable_state.documents.reference(rid)
            ref._score = score
            references.append(ref)

        return references

    def reference(self, element_id: int):
        return self._savable_state.documents.reference(element_id)

    def _get_text(self, result_id) -> str:
        return self._savable_state.documents.reference(result_id).text

    def text_to_result(self, text: str, result_id: int) -> None:
        """Trains NeuralDB to map the given text to the given entity ID.
        Also known as "upvoting".
        """
        teachers.upvote(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            query_id_para=[
                (text, upvote_id, self._get_text(result_id))
                for upvote_id in self._savable_state.documents.reference(
                    result_id
                ).upvote_ids
            ],
        )

    def text_to_result_batch(self, text_id_pairs: List[Tuple[str, int]]) -> None:
        """Trains NeuralDB to map the given texts to the given entity IDs.
        Also known as "batch upvoting".
        """
        query_id_para = [
            (query, upvote_id, self._get_text(result_id))
            for query, result_id in text_id_pairs
            for upvote_id in self._savable_state.documents.reference(
                result_id
            ).upvote_ids
        ]
        teachers.upvote(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            query_id_para=query_id_para,
        )

    def associate(self, source: str, target: str, strength: Strength = Strength.Strong):
        top_k = self._get_associate_top_k(strength)
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_pairs=[(source, target)],
            top_k=top_k,
        )

    def associate_batch(
        self, text_pairs: List[Tuple[str, str]], strength: Strength = Strength.Strong
    ):
        top_k = self._get_associate_top_k(strength)
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_pairs=text_pairs,
            top_k=top_k,
        )

    def _get_associate_top_k(self, strength):
        if strength == Strength.Weak:
            return 3
        elif strength == Strength.Medium:
            return 5
        elif strength == Strength.Strong:
            return 7
        else:
            return 7

    def supervised_train(
        self,
        data: List[Sup],
        learning_rate=0.0001,
        epochs=3,
    ):
        """Train on supervised datasets that correspond to specific sources.
        Suppose you inserted a "sports" product catalog and a "furniture"
        product catalog. You also have supervised datasets - pairs of queries
        and correct products - for both categories. You can use this method to
        train NeuralDB on these supervised datasets.
        """
        doc_manager = self._savable_state.documents
        query_col = self._savable_state.model.get_query_col()
        self._savable_state.model.get_model().train_on_data_source(
            data_source=SupDataSource(
                doc_manager=doc_manager,
                query_col=query_col,
                data=data,
                id_delimiter=self._savable_state.model.get_id_delimiter(),
            ),
            learning_rate=learning_rate,
            epochs=epochs,
        )

    def supervised_train_with_ref_ids(
        self,
        csv: str = None,
        query_column: str = None,
        id_column: str = None,
        id_delimiter: str = None,
        queries: Sequence[str] = None,
        labels: Sequence[Sequence[int]] = None,
        learning_rate=0.0001,
        epochs=3,
    ):
        """Train on supervised datasets that correspond to specific sources.
        Suppose you inserted a "sports" product catalog and a "furniture"
        product catalog. You also have supervised datasets - pairs of queries
        and correct products - for both categories. You can use this method to
        train NeuralDB on these supervised datasets.
        """
        doc_manager = self._savable_state.documents
        model_query_col = self._savable_state.model.get_query_col()
        self._savable_state.model.get_model().train_on_data_source(
            data_source=SupDataSource(
                doc_manager=doc_manager,
                query_col=model_query_col,
                data=[
                    Sup(
                        csv=csv,
                        query_column=query_column,
                        id_column=id_column,
                        id_delimiter=id_delimiter,
                        queries=queries,
                        labels=labels,
                        uses_db_id=True,
                    )
                ],
                id_delimiter=self._savable_state.model.get_id_delimiter(),
            ),
            learning_rate=learning_rate,
            epochs=epochs,
        )

    def get_associate_samples(self):
        """Get past associate() and associate_batch() samples from NeuralDB logs."""
        logs = self._savable_state.logger.get_logs()

        associate_logs = logs[logs["action"] == "associate"]
        associate_samples = []
        for _, row in associate_logs.iterrows():
            for source, target in row["args"]["pairs"]:
                associate_samples.append((source, target))

        return associate_samples

    def get_upvote_samples(self):
        """Get past text_to_result() and text_to_result_batch() samples from
        NeuralDB logs.
        """
        logs = self._savable_state.logger.get_logs()

        upvote_associate_samples = []
        upvote_logs = logs[logs["action"] == "upvote"]
        for _, row in upvote_logs.iterrows():
            if "query_id_para" in row["args"]:
                for source, _, target in row["args"]["query_id_para"]:
                    upvote_associate_samples.append((source, target))

        return upvote_associate_samples

    def get_rlhf_samples(self):
        """Get past associate(), associate_batch(), text_to_result(), and
        text_to_result_batch() samples from NeuralDB logs.
        """
        return self.get_associate_samples() + self.get_upvote_samples()

    def retrain(
        self,
        text_pairs: List[Tuple[str, str]] = [],
        learning_rate: float = 0.0001,
        epochs: int = 3,
        strength: Strength = Strength.Strong,
    ):
        """Train NeuralDB on all inserted documents and logged RLHF samples."""
        doc_manager = self._savable_state.documents

        if not text_pairs:
            text_pairs = self.get_rlhf_samples()

        self._savable_state.model.retrain(
            balancing_data=doc_manager.get_data_source(),
            source_target_pairs=text_pairs,
            n_buckets=self._get_associate_top_k(strength),
            learning_rate=learning_rate,
            epochs=epochs,
        )
