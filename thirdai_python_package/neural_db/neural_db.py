import copy
import shutil
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import thirdai
from thirdai._thirdai import bolt, data

from . import loggers, teachers
from .documents import CSV, Document, DocumentManager, Reference
from .models.finetunable_retriever import FinetunableRetriever
from .models.mach import Mach
from .models.mach_mixture_model import MachMixture
from .models.model_interface import CancelState
from .savable_state import (
    State,
    load_checkpoint,
    make_preinsertion_checkpoint,
    make_training_checkpoint,
)
from .supervised_datasource import Sup, SupDataSource
from .trainer.checkpoint_config import CheckpointConfig

Strength = Enum("Strength", ["Weak", "Medium", "Strong"])


def no_op(*args, **kwargs):
    pass


class NeuralDB:
    """
    NeuralDB is a search and retrieval system that can be used to search over
    knowledge bases and documents. It can also be used in RAG pipelines for the
    search retrieval phase.

    Examples:
        >>> ndb = NeuralDB()
        >>> ndb.insert([CSV(...), PDF(...), DOCX(...)])
        >>> results = ndb.search("how to make chocolate chip cookies")
    """

    def __init__(
        self,
        user_id: str = "user",
        num_shards: int = 1,
        num_models_per_shard: int = 1,
        retriever="finetunable_retriever",
        low_memory=None,
        **kwargs,
    ) -> None:
        """
        Constructs an empty NeuralDB.

        Args:
            user_id (str): Optional, used to identify user/session in logging.
            retriever (str): One of 'finetunable_retriever', 'mach', or 'hybrid'.
                Identifies which retriever to use as the backend. Defaults to
                'finetunable_retriever'.

        Returns:
            A NeuralDB.
        """
        if low_memory is not None:
            print(
                "Warning: 'low_memory' flag will be deprecated soon in the NeuralDB constructor. Please pass 'retriever=' instead."
            )
            if low_memory == True:
                retriever = "finetunable_retriever"
            elif low_memory == False:
                retriever = "hybrid"
        self._user_id: str = user_id

        # The savable_state kwarg is only used in static constructor methods
        # and should not be used by an external user.
        # We read savable_state from kwargs so that it doesn't appear in the
        # arguments list and confuse users.
        if "savable_state" not in kwargs:
            if num_shards <= 0:
                raise Exception(
                    f"Invalid Value Passed for num_shards : {num_shards}."
                    " NeuralDB can only be initialized with a positive number of"
                    " shards."
                )
            if num_models_per_shard <= 0:
                raise Exception(
                    f"Invalid Value Passed for num_models_per_shard : {num_models_per_shard}."
                    " NeuralDB can only be initialized with a positive number of"
                    " models per shard."
                )
            if retriever == "finetunable_retriever":
                model = FinetunableRetriever(**kwargs)
            elif retriever == "mach" or retriever == "hybrid":
                if num_shards > 1 or num_models_per_shard > 1:
                    model = MachMixture(
                        num_shards=num_shards,
                        num_models_per_shard=num_models_per_shard,
                        id_col="id",
                        query_col="query",
                        hybrid=(retriever == "hybrid"),
                        **kwargs,
                    )
                else:
                    model = Mach(
                        id_col="id",
                        query_col="query",
                        hybrid=(retriever == "hybrid"),
                        **kwargs,
                    )
            else:
                raise ValueError(
                    f"Invalid retriever '{retriever}'. Please use 'finetunable_retriever', 'mach', or 'hybrid'."
                )

            self._savable_state = State(
                model, logger=loggers.LoggerList([loggers.InMemoryLogger()])
            )
        else:
            self._savable_state = kwargs["savable_state"]

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        user_id: str = "user",
        on_progress: Callable = no_op,
        **kwargs,
    ):
        """
        Constructs a NeuralDB from a checkpoint. This can be used save and reload
        NeuralDBs, it is also used for loading pretrained NeuralDB models.

        Args:
            checkpoint_path (str): The path to the checkpoint directory.
            user_id (str): Optional, used to identify user/session in logging.
            on_progress (Callable): Optional, callback that can be called as loading the checkpoint progresses.

        Returns:
            A NeuralDB.
        """
        checkpoint_path = Path(checkpoint_path)
        savable_state = State.load(checkpoint_path, on_progress, **kwargs)
        if savable_state.model and savable_state.model.get_model():
            savable_state.model.set_mach_sampling_threshold(0.01)
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
        """
        Instantiate a NeuralDB, using the given UDT as the underlying model.
        Usually for porting a pretrained model into the NeuralDB format.
        Use the optional csv-related arguments to insert the pretraining dataset
        into the NeuralDB instance.

        Args:
            udt (bolt.UniversalDeepTransformer): The udt model to use in the NeuralDB.
            user_id (str): Optional, used to identify user/session in logging.
            csv (Optional[str]): Optional, default None. The path to the CSV file
                used to train the udt model. If supplied, the CSV file will be
                inserted into NeuralDB.
            csv_id_column (Optional[str]): Optional, default None. The id column
                of the training dataset. Required only if the data is being inserted via
                the `csv` arg.
            csv_strong_columns (Optional[str]): Optional, default None. The strong
                signal columns from the training data. Required only if the data is
                being inserted via the `csv` arg.
            csv_weak_columns (Optional[str]): Optional, default None. The weak signal
                columns from the training data. Required only if the data is being
                inserted via the `csv` arg.
            csv_reference_columns (Optional[str]): Optional, default None. The
                columns whose data should be returned as search results to queries.
                Required only if the data is being inserted via the `csv` arg.

        Returns:
            A NeuralDB.
        """
        if csv is None:
            udt.clear_index()

        udt.enable_rlhf()
        udt.set_mach_sampling_threshold(0.01)
        fhr, emb_dim, out_dim = udt.model_dims()

        text_dataset_config = udt.text_dataset_config()

        model = Mach(
            id_col=text_dataset_config.label_column,
            id_delimiter=text_dataset_config.label_delimiter,
            query_col=text_dataset_config.text_column,
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
                error_msg = (
                    "If the `csv` arg is provided, then the following args must also be"
                    " provided:\n"
                )
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
        if isinstance(self._savable_state.model, MachMixture):
            raise NotImplementedError(
                "Distributed Training is not supported for NeuralDB initialized with a"
                " mixture of experts."
            )
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
                "The pretrain_distributed function currently only supports CSV"
                " documents."
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

    def save(self, save_to: Union[str, Path], on_progress: Callable = no_op) -> str:
        if hasattr(self, "reranker_model"):
            delattr(self, "reranker_model")
        return self._savable_state.save(Path(save_to), on_progress)

    def _resume(
        self,
        on_progress: Callable,
        cancel_state: CancelState,
        checkpoint_config: CheckpointConfig,
        callbacks: List[bolt.train.callbacks.Callback] = None,
    ):
        documents, ids, resource_name = load_checkpoint(
            checkpoint_config=checkpoint_config
        )
        self._savable_state.documents = documents
        self._savable_state.model.resume(
            on_progress=on_progress,
            cancel_state=cancel_state,
            checkpoint_config=checkpoint_config.get_mach_config(),
            callbacks=callbacks,
        )

        return ids, resource_name

    def _insert_from_start(
        self,
        sources: List[Document],
        train: bool,
        fast_approximation: bool,
        num_buckets_to_sample: Optional[int],
        on_progress: Callable,
        on_error: Callable,
        cancel_state: CancelState,
        max_in_memory_batches: int,
        variable_length: Optional[data.transformations.VariableLengthConfig],
        checkpoint_config: CheckpointConfig,
        callbacks: List[bolt.train.callbacks.Callback] = None,
        **kwargs,
    ):
        documents_copy = copy.deepcopy(self._savable_state.documents)
        try:
            intro_and_train, ids = self._savable_state.documents.add(sources)
        except Exception as e:
            self._savable_state.documents = documents_copy
            if on_error is not None:
                on_error(error_msg=f"Failed to add files. {e.__str__()}")
                return []
            raise e

        if checkpoint_config:
            """
            We need to store the document manager state so that our label_id -> reference mapping remains consistent on resuming.
            """
            make_preinsertion_checkpoint(
                savable_state=self._savable_state,
                ids=ids,
                resource_name=intro_and_train.intro.resource_name(),
                checkpoint_config=checkpoint_config,
            )

        self._savable_state.model.index_from_start(
            intro_documents=intro_and_train.intro,
            train_documents=intro_and_train.train,
            num_buckets_to_sample=num_buckets_to_sample,
            fast_approximation=fast_approximation,
            should_train=train,
            on_progress=on_progress,
            cancel_state=cancel_state,
            max_in_memory_batches=max_in_memory_batches,
            variable_length=variable_length,
            checkpoint_config=(
                checkpoint_config.get_mach_config() if checkpoint_config else None
            ),
            callbacks=callbacks,
            **kwargs,
        )

        return ids, intro_and_train.intro.resource_name()

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
        max_in_memory_batches: int = None,
        variable_length: Optional[
            data.transformations.VariableLengthConfig
        ] = data.transformations.VariableLengthConfig(),
        checkpoint_config: Optional[CheckpointConfig] = None,
        callbacks: List[bolt.train.callbacks.Callback] = None,
        **kwargs,
    ) -> List[str]:
        """
        Inserts documents/resources into the database.

        Args:
            sources (List[Doc]): List of NeuralDB documents to be inserted.
            train (bool): Optional, defaults True. When True this means that the
                underlying model in the NeuralDB will undergo unsupervised pretraining
                on the inserted documents.
            fast_approximation (bool): Optional, default True. Much faster insertion
                with a slight drop in performance.
            num_buckets_to_sample (Optional[int]): Used to control load balacing when
                inserting entities into the NeuralDB.
            on_progress (Callable): Optional, a callback that is called at intervals
                as documents are inserted.
            on_success (Callable): Optional, a callback that is invoked when document
                insertion is finished successfully.
            on_error (Callable): Optional, a callback taht is invoked if an error occurs
                during insertion.
            cancel_state (CancelState): An object that can be used to stop an ongoing
                insertion. Primarily used for PocketLLM.
            max_in_memory_batches (int): Optional, default None. When supplied this limits
                the maximum amount of data that is loaded into memory at once during training.
                Useful for lower memory paradigms or with large datasets.
            checkpoint_config (CheckpointConfig): Optional, default None. Configuration for checkpointing during insertion. No checkpoints are created if checkpoint_config is unspecified.

        Returns:
            A list of the ids assigned to the inserted documents.
        """
        if checkpoint_config and checkpoint_config.resume_from_checkpoint:
            ids, resource_name = self._resume(
                on_progress=on_progress,
                cancel_state=cancel_state,
                checkpoint_config=checkpoint_config,
                callbacks=callbacks,
            )
        else:
            ids, resource_name = self._insert_from_start(
                sources=sources,
                train=train,
                fast_approximation=fast_approximation,
                num_buckets_to_sample=num_buckets_to_sample,
                on_progress=on_progress,
                on_error=on_error,
                cancel_state=cancel_state,
                max_in_memory_batches=max_in_memory_batches,
                variable_length=variable_length,
                checkpoint_config=checkpoint_config,
                callbacks=callbacks,
                **kwargs,
            )

        self._savable_state.logger.log(
            session_id=self._user_id,
            action="Train",
            args={"files": resource_name},
        )

        if checkpoint_config:
            # Once we have saved the model, we will delete the ndb checkpoint and save updated neural db with trained models.
            make_training_checkpoint(
                savable_state=self._savable_state, checkpoint_config=checkpoint_config
            )

        on_success()

        return ids

    def delete(self, source_ids: List[str]):
        """Deletes documents from the NeuralDB."""
        deleted_entities = self._savable_state.documents.delete(source_ids)
        self._savable_state.model.delete_entities(deleted_entities)
        self._savable_state.logger.log(
            session_id=self._user_id, action="delete", args={"source_ids": source_ids}
        )

    def clear_sources(self) -> None:
        """Removes all documents stored in the NeuralDB."""
        self._savable_state.documents.clear()
        self._savable_state.model.forget_documents()

    def _get_query_references(
        self,
        query: str,
        result_ids: List[Tuple[int, float, str]],
        top_k: int,
        rerank: bool,
        reranker: str,
        rerank_threshold,
        top_k_threshold,
    ):
        references = []
        for rid, score, retriever in result_ids:
            ref = self._savable_state.documents.reference(rid)
            ref._score = score
            ref._retriever = retriever
            references.append(ref)

        if rerank or reranker != None:
            if reranker is None:
                reranker = "semantic"

            keep, to_rerank = NeuralDB._split_references_for_reranking(
                references,
                rerank_threshold,
                average_top_k_scores=top_k_threshold if top_k_threshold else top_k,
            )

            reranked_indices, reranked_scores = self._rerank_references(
                query, to_rerank, reranker
            )

            reranked_scores = NeuralDB._scale_reranked_scores(
                original=[ref.score for ref in to_rerank],
                reranked=reranked_scores,
                leq=keep[-1].score if len(keep) > 0 else 1.0,
            )

            reranked = [to_rerank[i] for i in reranked_indices]
            for i, ref in enumerate(reranked):
                ref._score = reranked_scores[i]
            references = (keep + reranked)[:top_k]

        return references

    def _rerank_references(
        self,
        query,
        to_rerank,
        reranker,
    ):
        if len(to_rerank) == 0:
            return [], []

        if reranker == "semantic":
            # We add the reranked_model attribute because we don't want to load
            # the model on every search. Even though its cached its still slow.
            # We also make sure not to save the model with the neuraldb.
            if not hasattr(self, "reranker_model"):
                try:
                    from transformers import AutoModelForSequenceClassification
                except:
                    raise ValueError(
                        "Semantic reranking requires the 'transformers' package. Please run 'pip3 install transformers'"
                    )
                self.reranker_model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        "jinaai/jina-reranker-v1-tiny-en",
                        num_labels=1,
                        trust_remote_code=True,
                    )
                )
            scores = self.reranker_model.compute_score(
                [[query, ref.text] for ref in to_rerank]
            )
            reranked_indices = np.argsort(scores)[::-1]
            reranked_scores = np.sort(scores)[::-1]
        elif reranker == "lexical":
            ranker = thirdai.dataset.KeywordOverlapRanker()
            reranked_indices, reranked_scores = ranker.rank(
                query, [ref.text for ref in to_rerank]
            )
        else:
            raise ValueError(
                "Invalid argument for reranker. Options are 'semantic' and 'lexical'"
            )

        return reranked_indices, reranked_scores

    @staticmethod
    def _split_references_for_reranking(
        references,
        rerank_threshold,
        average_top_k_scores,
    ):
        if rerank_threshold is None:
            rerank_start = 0
        else:
            scores = np.array([ref.score for ref in references])
            mean_score = np.mean(scores[:average_top_k_scores])
            rerank_start = np.searchsorted(
                -scores, -rerank_threshold * mean_score, side="right"
            )
        return references[:rerank_start], references[rerank_start:]

    @staticmethod
    def _scale_reranked_scores(
        original: List[float], reranked: List[float], leq: float
    ):
        """The scores returned by the reranker are not in the same scale as
        the original score. To fix this, transform the reranked scores such that
        they are in the same range as the original scores.
        """
        if len(original) == 0:
            return []
        reranked_delta = reranked[0] - reranked[-1]
        if reranked_delta == 0:
            return [original[0] for _ in reranked]
        original_delta = original[0] - original[-1]
        delta_scaler = original_delta / reranked_delta
        return [
            original[-1] + (score - reranked[-1]) * delta_scaler for score in reranked
        ]

    def search(
        self,
        query: str,
        top_k: int,
        constraints=None,
        rerank=False,
        reranker=None,
        top_k_rerank=100,
        rerank_threshold=1.5,
        top_k_threshold=None,
        retriever=None,
        label_probing=False,
        mach_first=False,
    ) -> List[Reference]:
        """
        Searches the contents of the NeuralDB for documents relevant to the given query.

        Args:
            query (str): The query to search with.
            top_k (int): The number of results to return.
            constraints (Dict[str, Any]): A dictionary containing constraints to
                apply to the metadata field of each document in the NeuralDB. This
                allows for queries that will only return results with a certain property.
                The constrains are in the form {"metadata_key": <constraint>} where
                <constraint> is either an explicit value for the key in the metadata,
                or a Filter object.
            rerank (bool): Optional, default False. When True an additional reranking
                step is applied to results.
            top_k_rerank (int): Optional, default 100. If rerank=True then this argument
                determines how many candidates are retrieved, before reranking and
                returning the top_k.
            rerank_threshold (float): Optional, default 1.5. In reranking all candidates
                with a score under a certain threshold are reranked. This threshold
                is computed as this argument (`rerank_threshold`) times the average score
                over the first top_k_threshold candidates. Candidates with scores lower
                than this threshold will be reranked. Thus, increasing this value
                causes more candidates to be reranked.
            top_k_threshold (Optional[float]): Optional, default None, which means
                the arg `top_k` will be used. If specified this argument controls
                how many of the top candidates' scores are averaged to obtain the
                mean that is used to determine which candidates are reranked. For
                example passing rerank_threshold=2 and top_k_threshold=4 means that
                the scores of the top 4 elements are averaged, and all elements below
                2x this average are reranked.
            retriever (Optional[str]): Optional, default None. This arg controls which
                retriever to use for search when a hybrid retrieval model is used. Passing
                None means that NeuralDB will automatically decide which retrievers (or
                combination of retrievers) to use.

        Returns:
            List[Reference]: A list of Reference objects. Each reference object contains text data matching
            the query, along with information about which document contained that text.

        Examples:
            >>> ndb.search("what is ...", top_k=5)
            >>> ndb.search("what is ...", top_k=5, constraints={"file_type": "pdf", "file_created", GreaterThan(10)})
        """
        return self.search_batch(
            queries=[query],
            top_k=top_k,
            constraints=constraints,
            rerank=rerank,
            reranker=reranker,
            top_k_rerank=top_k_rerank,
            rerank_threshold=rerank_threshold,
            top_k_threshold=top_k_threshold,
            retriever=retriever,
            label_probing=label_probing,
            mach_first=mach_first,
        )[0]

    def search_batch(
        self,
        queries: List[str],
        top_k: int,
        constraints=None,
        rerank=False,
        reranker=None,
        top_k_rerank=100,
        rerank_threshold=1.5,
        top_k_threshold=None,
        retriever=None,
        label_probing=False,
        mach_first=False,
    ):
        """
        Runs search on a batch of queries for much faster throughput.

        Args:
            queries (List[str]): The queries to search.

        Returns:
            List[List[Reference]]: Combines each result of db.search into a list.
        """
        if rerank and top_k_rerank < top_k:
            raise ValueError("top_k_rerank should not be smaller than top_k.")
        matching_entities = None
        top_k_to_search = top_k_rerank if rerank else top_k
        if constraints:
            matching_entities = self._savable_state.documents.entity_ids_by_constraints(
                constraints
            )
            queries_result_ids = self._savable_state.model.score(
                samples=queries,
                entities=[matching_entities] * len(queries),
                n_results=top_k_to_search,
            )
        else:
            queries_result_ids = self._savable_state.model.infer_labels(
                samples=queries,
                n_results=top_k_to_search,
                retriever="mach" if rerank else retriever,
                label_probing=label_probing,
                mach_first=mach_first,
            )

        return [
            self._get_query_references(
                query,
                result_ids,
                top_k,
                rerank,
                reranker,
                rerank_threshold,
                top_k_threshold,
            )
            for query, result_ids in zip(queries, queries_result_ids)
        ]

    def reference(self, element_id: int):
        """Returns a reference containing the text and other information for a given entity id."""
        return self._savable_state.documents.reference(element_id)

    def _get_text(self, result_id) -> str:
        return self._savable_state.documents.reference(result_id).text

    def text_to_result(self, text: str, result_id: int, **kwargs) -> None:
        """Trains NeuralDB to map the given text to the given entity ID.
        Also known as "upvoting".

        Example:
            >>> ndb.text_to_result("a new query", result_id=4)
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
            **kwargs,
        )

    def text_to_result_batch(
        self, text_id_pairs: List[Tuple[str, int]], **kwargs
    ) -> None:
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
            **kwargs,
        )

    def associate(
        self, source: str, target: str, strength: Strength = Strength.Strong, **kwargs
    ):
        """
        Teaches the underlying model in the NeuralDB that two different texts
        correspond to similar concepts or queries.

        Args:
            source (str): The source is the new text you want to teach the model about.
            target (str): The target is the known text that is provided to the model
                as an example of the type of information or query the source resembles.

        Examples:
            >>> ndb.associate("asap", "as soon as possible")
            >>> ndb.associate("what is a 401k", "explain different types of retirement savings")
        """
        top_k = self._get_associate_top_k(strength)
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_pairs=[(source, target)],
            top_k=top_k,
            **kwargs,
        )

    def associate_batch(
        self,
        text_pairs: List[Tuple[str, str]],
        strength: Strength = Strength.Strong,
        **kwargs,
    ):
        """Same as associate, but the process is applied to a batch of (source, target) pairs at once."""
        top_k = self._get_associate_top_k(strength)
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_pairs=text_pairs,
            top_k=top_k,
            **kwargs,
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
        batch_size: Optional[int] = None,
        max_in_memory_batches: Optional[int] = None,
        metrics: List[str] = [],
        callbacks: List[bolt.train.callbacks.Callback] = [],
        checkpoint_config: Optional[CheckpointConfig] = None,
        **kwargs,
    ):
        """
        Train on supervised datasets that correspond to specific sources.
        Suppose you inserted a "sports" product catalog and a "furniture"
        product catalog. You also have supervised datasets - pairs of queries
        and correct products - for both categories. You can use this method to
        train NeuralDB on these supervised datasets.

        Args:
            data (List[Sup]): Supervised training samples.
            learning_rate (float): Optional. The learning rate to use for training.
            epochs (int): Optional. The number of epochs to train for.
        """
        doc_manager = self._savable_state.documents
        query_col = self._savable_state.model.get_query_col()
        self._savable_state.model.train_on_supervised_data_source(
            supervised_data_source=SupDataSource(
                doc_manager=doc_manager,
                query_col=query_col,
                data=data,
                id_delimiter=self._savable_state.model.get_id_delimiter(),
            ),
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            max_in_memory_batches=max_in_memory_batches,
            metrics=metrics,
            callbacks=callbacks,
            disable_finetunable_retriever=kwargs.get(
                "disable_finetunable_retriever", True
            ),
            checkpoint_config=checkpoint_config,
        )

        if checkpoint_config:
            make_training_checkpoint(self._savable_state, checkpoint_config)

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
        batch_size: Optional[int] = None,
        max_in_memory_batches: Optional[int] = None,
        metrics: List[str] = [],
        callbacks: List[bolt.train.callbacks.Callback] = [],
        checkpoint_config: Optional[CheckpointConfig] = None,
        **kwargs,
    ):
        """Train on supervised datasets that correspond to specific sources.
        Suppose you inserted a "sports" product catalog and a "furniture"
        product catalog. You also have supervised datasets - pairs of queries
        and correct products - for both categories. You can use this method to
        train NeuralDB on these supervised datasets. This method must be invoked
        with either A) a csv file with the query and id columns within it, or B) an
        explicit list of queries and expected labels.
        """
        doc_manager = self._savable_state.documents
        model_query_col = self._savable_state.model.get_query_col()
        self._savable_state.model.train_on_supervised_data_source(
            supervised_data_source=SupDataSource(
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
            batch_size=batch_size,
            max_in_memory_batches=max_in_memory_batches,
            metrics=metrics,
            callbacks=callbacks,
            disable_finetunable_retriever=kwargs.get(
                "disable_finetunable_retriever", True
            ),
            checkpoint_config=checkpoint_config,
        )
        if checkpoint_config:
            make_training_checkpoint(self._savable_state, checkpoint_config)

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
