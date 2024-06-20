from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

from .bazaar_base import Bazaar, auth_header
from .utils import (
    check_deployment_decorator,
    construct_deployment_url,
    create_deployment_identifier,
    create_model_identifier,
    http_get_with_error,
    http_post_with_error,
    print_progress_dots,
)


class Model:
    """
    A class representing a model listed on NeuralDB Enterprise.

    Attributes:
        _model_identifier (str): The unique identifier for the model.

    Methods:
        __init__(self, model_identifier: str) -> None:
            Initializes a new instance of the Model class.

            Parameters:
                model_identifier (str): An optional model identifier.

        model_identifier(self) -> str:
            Getter method for accessing the model identifier.

            Returns:
                str: The model identifier, or None if not set.
    """

    def __init__(self, model_identifier, model_id=None) -> None:
        self._model_identifier = model_identifier
        self._model_id = model_id

    @property
    def model_identifier(self):
        return self._model_identifier

    @property
    def model_id(self):
        if self._model_id:
            return self._model_id
        raise ValueError("Model id is not yet set.")


class NeuralDBClient:
    """
    A client for interacting with the deployed NeuralDB model.

    Attributes:
        deployment_identifier (str): The identifier for the deployment.
        deployment_id (str): The deployment ID for the deployed NeuralDB model.
        bazaar (thirdai.neural_db.ModelBazaar): The bazaar object corresponding to a NeuralDB Enterprise installation

    Methods:
        __init__(self, deployment_identifier: str, deployment_id: str, bazaar: ModelBazaar) -> None:
            Initializes a new instance of the NeuralDBClient.

        search(self, query, top_k=5, constraints: Optional[dict[str, dict[str, str]]]=None) -> List[dict]:
            Searches the ndb model for relevant search results.

        insert(self, documents: list[dict[str, Any]]) -> None:
            Inserts documents into the ndb model.

        delete(self, source_ids: List[str]) -> None:
            Deletes documents from the ndb model

        associate(self, text_pairs (List[Dict[str, str]])) -> None:
            Associates source and target string pairs in the ndb model.

        upvote(self, text_id_pairs: List[Dict[str, Union[str, int]]]) -> None:
            Upvotes a response in the ndb model.

        downvote(self, text_id_pairs: List[Dict[str, Union[str, int]]]) -> None:
            Downvotes a response in the ndb model.

        chat(self, user_input: str, session_id: str) -> Dict[str, str]:
            Returns a reply given the user_input and the chat history associated with session_id

        get_chat_history(self, session_id: str) -> Dict[List[Dict[str, str]]]:
            Returns chat history associated with session_id

        sources(self) -> List[Dict[str, str]]:
            Gets the source names and ids of documents in the ndb model
    """

    def __init__(
        self, deployment_identifier: str, deployment_id: str, bazaar: ModelBazaar
    ):
        """
        Initializes a new instance of the NeuralDBClient.

        Args:
            deployment_identifier (str): The identifier for the deployment.
            deployment_id (str): The deployment ID for the deployed NeuralDB model.
            bazaar (thirdai.neural_db.ModelBazaar): The bazaar object corresponding to a NeuralDB Enterprise installation
        """
        self.deployment_identifier = deployment_identifier
        self.base_url = construct_deployment_url(
            re.sub(r"api/$", "", bazaar._base_url), deployment_id
        )
        self.bazaar = bazaar

    @check_deployment_decorator
    def search(
        self, query, top_k=5, constraints: Optional[dict[str, dict[str, str]]] = None
    ):
        """
        Searches the ndb model for similar queries.

        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to retrieve (default is 10).
            constraints (Optional[dict[str, dict[str, str]]]): Constraints to filter the search result metadata by.
                These constraints must be in the following format:
                {"FIELD_NAME": {"constraint_type": "CONSTRAINT_NAME", **kwargs}} where
                "FIELD_NAME" is the field that you want to filter over, and "CONSTRAINT_NAME"
                is one of the following: "AnyOf", "EqualTo", "InRange", "GreaterThan", and "LessThan".
                The kwargs for the above constraints are shown below:

                class AnyOf(BaseModel):
                    constraint_type: Literal["AnyOf"]
                    values: Iterable[Any]

                class EqualTo(BaseModel):
                    constraint_type: Literal["EqualTo"]
                    value: Any

                class InRange(BaseModel):
                    constraint_type: Literal["InRange"]
                    minimum: Any
                    maximum: Any
                    inclusive_min: bool = True
                    inclusive_max: bool = True

                class GreaterThan(BaseModel):
                    constraint_type: Literal["GreaterThan"]
                    minimum: Any
                    include_equal: bool = False

                class LessThan(BaseModel):
                    constraint_type: Literal["LessThan"]
                    maximum: Any
                    include_equal: bool = False

        Returns:
            Dict: A dict of search results containing keys: `query_text` and `references`.
        """

        response = http_post_with_error(
            urljoin(self.base_url, "predict"),
            params={"query_text": query, "top_k": top_k},
            json=constraints,
            headers=auth_header(self.bazaar._access_token),
        )

        return json.loads(response.content)["data"]

    @check_deployment_decorator
    def insert(self, documents: list[dict[str, Any]]):
        """
        Inserts documents into the ndb model.

        Args:
            documents (List[dict[str, Any]]): A list of dictionaries that represent documents to be inserted to the ndb model.
                The document dictionaries must be in the following format:
                {"document_type": "DOCUMENT_TYPE", **kwargs} where "DOCUMENT_TYPE" is one of the following:
                "PDF", "CSV", "DOCX", "URL", "SentenceLevelPDF", "SentenceLevelDOCX", "Unstructured", "InMemoryText".
                The kwargs for each document type are shown below:

                class PDF(Document):
                    document_type: Literal["PDF"]
                    path: str
                    metadata: Optional[dict[str, Any]] = None
                    on_disk: bool = False
                    version: str = "v1"
                    chunk_size: int = 100
                    stride: int = 40
                    emphasize_first_words: int = 0
                    ignore_header_footer: bool = True
                    ignore_nonstandard_orientation: bool = True

                class CSV(Document):
                    document_type: Literal["CSV"]
                    path: str
                    id_column: Optional[str] = None
                    strong_columns: Optional[List[str]] = None
                    weak_columns: Optional[List[str]] = None
                    reference_columns: Optional[List[str]] = None
                    save_extra_info: bool = True
                    metadata: Optional[dict[str, Any]] = None
                    has_offset: bool = False
                    on_disk: bool = False

                class DOCX(Document):
                    document_type: Literal["DOCX"]
                    path: str
                    metadata: Optional[dict[str, Any]] = None
                    on_disk: bool = False

                class URL(Document):
                    document_type: Literal["URL"]
                    url: str
                    save_extra_info: bool = True
                    title_is_strong: bool = False
                    metadata: Optional[dict[str, Any]] = None
                    on_disk: bool = False

                class SentenceLevelPDF(Document):
                    document_type: Literal["SentenceLevelPDF"]
                    path: str
                    metadata: Optional[dict[str, Any]] = None
                    on_disk: bool = False

                class SentenceLevelDOCX(Document):
                    document_type: Literal["SentenceLevelDOCX"]
                    path: str
                    metadata: Optional[dict[str, Any]] = None
                    on_disk: bool = False

                class Unstructured(Document):
                    document_type: Literal["Unstructured"]
                    path: str
                    save_extra_info: bool = True
                    metadata: Optional[dict[str, Any]] = None
                    on_disk: bool = False

                class InMemoryText(Document):
                    document_type: Literal["InMemoryText"]
                    name: str
                    texts: list[str]
                    metadatas: Optional[list[dict[str, Any]]] = None
                    global_metadata: Optional[dict[str, Any]] = None
                    on_disk: bool = False

                For Document types with the arg "path", ensure that the path exists on your local machine.
        """

        if not documents:
            raise ValueError("Documents cannot be empty.")

        files = []
        for doc in documents:
            if "path" in doc and ("location" not in doc or doc["location"] == "local"):
                if not os.path.exists(doc["path"]):
                    raise ValueError(
                        f"Path {doc['path']} was provided but doesn't exist on the machine."
                    )
                files.append(("files", open(doc["path"], "rb")))

        files.append(("documents", (None, json.dumps(documents), "application/json")))

        response = http_post_with_error(
            urljoin(self.base_url, "insert"),
            files=files,
            headers=auth_header(self.bazaar._access_token),
        )

        return json.loads(response.content)["data"]

    @check_deployment_decorator
    def ainsert(self, documents: list[dict[str, Any]]):
        """
        Inserts documents into the ndb model asynchronously.

        Args: Look at insert() for args.

        Returns:
            data (dict[str, str]): A dict containing the task id for the insertion

        """

        if not documents:
            raise ValueError("Documents cannot be empty.")

        files = []
        for doc in documents:
            if "path" in doc and ("location" not in doc or doc["location"] == "local"):
                if not os.path.exists(doc["path"]):
                    raise ValueError(
                        f"Path {doc['path']} was provided but doesn't exist on the machine."
                    )
                files.append(("files", open(doc["path"], "rb")))

        files.append(("documents", (None, json.dumps(documents), "application/json")))

        response = http_post_with_error(
            urljoin(self.base_url, "ainsert"),
            files=files,
            headers=auth_header(self.bazaar._access_token),
        )

        return json.loads(response.content)["data"]

    @check_deployment_decorator
    def task_status(self, task_id: str):
        """
        Gets the task for the given task_id

        Args:
            task_id (str): A task id

        """

        response = http_post_with_error(
            urljoin(self.base_url, "task-status"),
            params={"task_id": task_id},
            headers=auth_header(self.bazaar._access_token),
        )

        return json.loads(response.content)["data"]

    @check_deployment_decorator
    def delete(self, source_ids: List[str]):
        """
        Deletes documents from the ndb model using source ids.

        Args:
            files (List[str]): A list of source ids to delete from the ndb model.
        """
        response = http_post_with_error(
            urljoin(self.base_url, "delete"),
            json={"source_ids": source_ids},
            headers=auth_header(self.bazaar._access_token),
        )

    @check_deployment_decorator
    def associate(self, text_pairs: List[Dict[str, str]]):
        """
        Associates source and target string pairs in the ndb model.

        Args:
            text_pairs (List[Dict[str, str]]): List of dictionaries where each dictionary has 'source' and 'target' keys.
        """
        response = http_post_with_error(
            urljoin(self.base_url, "associate"),
            json={"text_pairs": text_pairs},
            headers=auth_header(self.bazaar._access_token),
        )

    @check_deployment_decorator
    def save_model(self, override: bool = True, model_name: Optional[str] = None):

        response = http_post_with_error(
            urljoin(self.base_url, "save"),
            json={"override": override, "model_name": model_name},
            headers=auth_header(self.bazaar._access_token),
        )

        print("Successfully saved the model.")

        content = response.json()["data"]

        if content["new_model_id"]:
            return Model(
                model_identifier=create_model_identifier(
                    model_name, self.bazaar._username
                ),
                model_id=content["new_model_id"],
            )

        return None

    @check_deployment_decorator
    def upvote(self, text_id_pairs: List[Dict[str, Union[str, int]]]):
        """
        Upvote response with 'reference_id' corresponding to 'query_text' in the ndb model.

        Args:
            text_id_pairs: (List[Dict[str, Union[str, int]]]): List of dictionaries where each dictionary has 'query_text' and 'reference_id' keys.
        """
        response = http_post_with_error(
            urljoin(self.base_url, "upvote"),
            json={"text_id_pairs": text_id_pairs},
            headers=auth_header(self.bazaar._access_token),
        )

        print("Successfully upvoted the specified search result.")

    @check_deployment_decorator
    def downvote(self, text_id_pairs: List[Dict[str, Union[str, int]]]):
        """
        Downvote response with 'reference_id' corresponding to 'query_text' in the ndb model.

        Args:
            text_id_pairs: (List[Dict[str, Union[str, int]]]): List of dictionaries where each dictionary has 'query_text' and 'reference_id' keys.
        """
        response = http_post_with_error(
            urljoin(self.base_url, "downvote"),
            json={"text_id_pairs": text_id_pairs},
            headers=auth_header(self.bazaar._access_token),
        )

        print("Successfully downvoted the specified search result.")

    @check_deployment_decorator
    def chat(self, user_input: str, session_id: str) -> Dict[str, str]:
        """
        Returns a reply given the user_input and the chat history associated with session_id

        Args:
            user_input (str): The user input for the chatbot to respond to
            session_id (str): The session id corresponding to a specific chat session
        """
        response = http_post_with_error(
            urljoin(self.base_url, "chat"),
            json={"user_input": user_input, "session_id": session_id},
            headers=auth_header(self.bazaar._access_token),
        )

        return response.json()["data"]

    @check_deployment_decorator
    def get_chat_history(self, session_id: str) -> Dict[List[Dict[str, str]]]:
        """
        Returns chat history associated with session_id

        Args:
            session_id (str): The session id corresponding to a specific chat session
        """
        response = http_post_with_error(
            urljoin(self.base_url, "get-chat-hisory"),
            json={"session_id": session_id},
            headers=auth_header(self.bazaar._access_token),
        )

        return response.json()["data"]

    @check_deployment_decorator
    def sources(self) -> List[Dict[str, str]]:
        """
        Gets the source names and ids of documents in the ndb model

        """
        response = http_get_with_error(
            urljoin(self.base_url, "sources"),
            headers=auth_header(self.bazaar._access_token),
        )

        return response.json()["data"]


class ModelBazaar(Bazaar):
    """
    A class representing ModelBazaar, providing functionality for managing models and deployments.

    Attributes:
        _base_url (str): The base URL for the Model Bazaar.
        _cache_dir (Union[Path, str]): The directory for caching downloads.

    Methods:
        __init__(self, base_url: str, cache_dir: Union[Path, str] = "./bazaar_cache") -> None:
            Initializes a new instance of the ModelBazaar class.

        sign_up(self, email: str, password: str, username: str) -> None:
            Signs up a user and sets the username for the ModelBazaar instance.

        log_in(self, email: str, password: str) -> None:
            Logs in a user and sets user-related attributes for the ModelBazaar instance.

        push_model(self, model_name: str, local_path: str, access_level: str = "private") -> None:
            Pushes a model to the Model Bazaar.

        pull_model(self, model_identifier: str) -> NeuralDBClient:
            Pulls a model from the Model Bazaar and returns a NeuralDBClient instance.

        list_models(self) -> List[dict]:
            Lists available models in the Model Bazaar.

        train(self,
            model_name: str,
            unsupervised_docs: Optional[List[str]] = None,
            supervised_docs: Optional[List[Tuple[str, str]]] = None,
            test_doc: Optional[str] = None,
            doc_type: str = "local",
            sharded: bool = False,
            is_async: bool = False,
            base_model_identifier: str = None,
            train_extra_options: Optional[dict] = None,
            metadata: Optional[List[Dict[str, str]]] = None
        ) -> Model:
            Initiates training for a model and returns a Model instance.

        await_train(self, model: Model) -> None:
            Waits for the training of a model to complete.

        test(self,
            model_identifier: str,
            test_doc: str,
            doc_type: str = "local",
            test_extra_options: dict = {},
            is_async: bool = False,
        ) -> str:
            Starts the Model testing on given test file.

        await_test(self, model_identifier: str, test_id: str) -> None:
            Waits for the testing of a model on that test_id to complete.

        deploy(self, model_identifier: str, deployment_name: str, is_async: bool = False) -> NeuralDBClient:
            Deploys a model and returns a NeuralDBClient instance.

        await_deploy(self, ndb_client: NeuralDBClient) -> None:
            Waits for the deployment of a model to complete.

        undeploy(self, ndb_client: NeuralDBClient) -> None:
            Undeploys a deployed model.

        list_deployments(self) -> List[dict]:
            Lists the deployments in the Model Bazaar.

        connect(self, deployment_identifier: str) -> NeuralDBClient:
            Connects to a deployed model and returns a NeuralDBClient instance.
    """

    def __init__(
        self,
        base_url: str,
        cache_dir: Union[Path, str] = "./bazaar_cache",
    ):
        """
        Initializes a new instance of the ModelBazaar class.

        Args:
            base_url (str): The base URL for the Model Bazaar.
            cache_dir (Union[Path, str]): The directory for caching downloads.
        """
        super().__init__(base_url, cache_dir)
        self._username = None
        self._access_token = None
        self._doc_types = ["local", "nfs", "s3"]

    def sign_up(self, email, password, username):
        """
        Signs up a user and sets the username for the ModelBazaar instance.

        Args:
            email (str): The email of the user.
            password (str): The password of the user.
            username (str): The desired username.
        """
        self.signup(email=email, password=password, username=username)
        self._username = username

    def log_in(self, email, password):
        """
        Logs in a user and sets user-related attributes for the ModelBazaar instance.

        Args:
            email (str): The email of the user.
            password (str): The password of the user.
        """
        self.login(email=email, password=password)
        self._access_token = self._login_instance.access_token
        self._username = self._login_instance.username

    def push_model(
        self, model_name: str, local_path: str, access_level: str = "private"
    ):
        """
        Pushes a model to the Model Bazaar.

        Args:
            model_name (str): The name of the model.
            local_path (str): The local path of the model.
            access_level (str): The access level for the model (default is "private").
        """
        self.push(
            name=model_name,
            model_path=local_path,
            trained_on="Own Documents",
            access_level=access_level,
            is_indexed=True,
            description="",
        )

    def pull_model(self, model_identifier: str):
        """
        Pulls a model from the Model Bazaar and returns a NeuralDBClient instance.

        Args:
            model_identifier (str): The identifier of the model.

        Returns:
            NeuralDBClient: A NeuralDBClient instance.
        """
        return self.get_neuraldb(model_identifier=model_identifier)

    def list_models(self):
        """
        Lists available models in the Model Bazaar.

        Returns:
            List[dict]: A list of dictionaries containing information about available models.
        """
        return self.fetch()

    def train(
        self,
        model_name: str,
        unsupervised_docs: Optional[List[str]] = None,
        supervised_docs: Optional[List[Tuple[str, str]]] = None,
        test_doc: Optional[str] = None,
        doc_type: str = "local",
        sharded: bool = False,
        is_async: bool = False,
        base_model_identifier: Optional[str] = None,
        train_extra_options: Optional[dict] = None,
        metadata: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initiates training for a model and returns a Model instance.

        Args:
            model_name (str): The name of the model.
            unsupervised_docs (Optional[List[str]]): A list of document paths for unsupervised training.
            supervised_docs (Optional[List[Tuple[str, str]]]): A list of document path and source id pairs.
            test_doc (Optional[str]): A path to a test file for evaluating the trained NeuralDB.
            doc_type (str): Specifies document location type : "local"(default), "nfs" or "s3".
            sharded (bool): Whether NeuralDB training will be distributed over NeuralDB shards.
            is_async (bool): Whether training should be asynchronous (default is False).
            train_extra_options: (Optional[dict])
            base_model_identifier (Optional[str]): The identifier of the base model.
            metadata (Optional[List[Dict[str, str]]]): A list metadata dicts. Each dict corresponds to an unsupervised file.

        Returns:
            Model: A Model instance.
        """
        if doc_type not in self._doc_types:
            raise ValueError(
                f"Invalid doc_type value. Supported doc_type are {self._doc_types}"
            )

        if not unsupervised_docs and not supervised_docs:
            raise ValueError("Both the unsupervised and supervised docs are empty.")

        if metadata and unsupervised_docs:
            if len(metadata) != len(unsupervised_docs):
                raise ValueError("Metadata is not provided for all unsupervised files.")

        file_details_list = []
        docs = []

        if unsupervised_docs and metadata:
            for doc, meta in zip(unsupervised_docs, metadata):
                docs.append(doc)
                file_details_list.append(
                    {"mode": "unsupervised", "location": doc_type, "metadata": meta}
                )
        elif unsupervised_docs:
            for doc in unsupervised_docs:
                docs.append(doc)
                file_details_list.append({"mode": "unsupervised", "location": doc_type})

        if supervised_docs:
            for sup_file, source_id in supervised_docs:
                docs.append(sup_file)
                file_details_list.append(
                    {"mode": "supervised", "location": doc_type, "source_id": source_id}
                )

        if test_doc:
            docs.append(test_doc)
            file_details_list.append({"mode": "test", "location": doc_type})

        url = urljoin(self._base_url, f"jobs/train")
        files = [
            (
                ("files", open(file_path, "rb"))
                if doc_type == "local"
                else ("files", (file_path, "don't care"))
            )
            for file_path in docs
        ]
        if train_extra_options:
            files.append(
                (
                    "extra_options_form",
                    (None, json.dumps(train_extra_options), "application/json"),
                )
            )

        files.append(
            (
                "file_details_list",
                (
                    None,
                    json.dumps({"file_details": file_details_list}),
                    "application/json",
                ),
            )
        )

        response = http_post_with_error(
            url,
            params={
                "model_name": model_name,
                "doc_type": doc_type,
                "sharded": sharded,
                "base_model_identifier": base_model_identifier,
            },
            files=files,
            headers=auth_header(self._access_token),
        )
        print(response.content)
        response_content = json.loads(response.content)
        if response_content["status"] != "success":
            raise Exception(response_content["message"])

        model = Model(
            model_identifier=create_model_identifier(
                model_name=model_name, author_username=self._username
            ),
            model_id=response_content["data"]["model_id"],
        )

        if is_async:
            return model

        self.await_train(model)
        return model

    def test(
        self,
        model_identifier: str,
        test_doc: str,
        doc_type: str = "local",
        test_extra_options: dict = {},
        is_async: bool = False,
    ):
        """
        Initiates testing for a model and returns the test_id (unique identifier for this test)

        Args:
            model_identifier (str): The identifier of the model.
            test_doc (str): A path to a test file for evaluating the trained NeuralDB.
            doc_type (str): Specifies document location type : "local"(default), "nfs" or "s3".
            test_extra_options: (Optional[dict])
            is_async (bool): Whether testing should be asynchronous (default is False).

        Returns:
            str: The test_id which is unique for given testing.
        """
        url = urljoin(self._base_url, f"test/test")

        files = [
            (
                ("file", open(test_doc, "rb"))
                if doc_type == "local"
                else ("file", (test_doc, "don't care"))
            )
        ]
        if test_extra_options:
            files.append(
                (
                    "extra_options_form",
                    (None, json.dumps(test_extra_options), "application/json"),
                )
            )

        response = http_post_with_error(
            url,
            params={
                "doc_type": doc_type,
                "model_identifier": model_identifier,
            },
            files=files,
            headers=auth_header(self._access_token),
        )
        print(response.content)

        response_content = json.loads(response.content)
        if response_content["status"] != "success":
            raise Exception(response_content["message"])

        if is_async:
            return response_content["data"]["data_id"]

        self.await_test(model_identifier, response_content["data"]["data_id"])
        return response_content["data"]["data_id"]

    def test_status(self, test_id: str):
        """
        Checks for the status of the model testing

        Args:
            test_id (str): The unique id with which we can recognize the test,
            the user will get this id in the response when they trigger the test.
        """

        url = urljoin(self._base_url, f"test/test-status")

        response = http_get_with_error(
            url,
            params={"test_id": test_id},
            headers=auth_header(self._access_token),
        )

        response_data = json.loads(response.content)["data"]

        return response_data

    def await_test(self, model_identifier: str, test_id: str):
        """
        Waits for the testing of the model to complete.

        Args:
            model_identifier: The identifier of the model.
            test_id: Unique id for the test.
        """

        while True:
            response_data = self.test_status(test_id)

            if response_data["status"] == "complete":
                print("\nTesting completed")
                return response_data["results"]

            if response_data["status"] == "failed":
                print("\nTesting Failed")
                raise ValueError(f"Test Failed for {model_identifier} and {test_id}")

            print("Testing: In progress", end="", flush=True)
            print_progress_dots(duration=10)

    def train_status(self, model: Model):
        """
        Checks for the status of the model training

        Args:
            model (Model): The Model instance.
        """

        url = urljoin(self._base_url, f"jobs/train-status")

        response = http_get_with_error(
            url,
            params={"model_identifier": model.model_identifier},
            headers=auth_header(self._access_token),
        )

        response_data = json.loads(response.content)["data"]

        return response_data

    def await_train(self, model: Model):
        """
        Waits for the training of a model to complete.

        Args:
            model (Model): The Model instance.
        """
        while True:
            response_data = self.train_status(model)

            if response_data["status"] == "complete":
                print("\nTraining completed")
                return

            if response_data["status"] == "failed":
                print("\nTraining Failed")
                raise ValueError(f"Training Failed for {model.model_identifier}")

            print("Training: In progress", end="", flush=True)
            print_progress_dots(duration=10)

    def deploy(
        self,
        model_identifier: str,
        deployment_name: str,
        memory: Optional[int] = None,
        is_async=False,
    ):
        """
        Deploys a model and returns a NeuralDBClient instance.

        Args:
            model_identifier (str): The identifier of the model.
            deployment_name (str): The name for the deployment.
            is_async (bool): Whether deployment should be asynchronous (default is False).

        Returns:
            NeuralDBClient: A NeuralDBClient instance.
        """
        url = urljoin(self._base_url, f"jobs/deploy")
        params = {
            "model_identifier": model_identifier,
            "deployment_name": deployment_name,
            "memory": memory,
        }
        response = http_post_with_error(
            url, params=params, headers=auth_header(self._access_token)
        )
        response_data = json.loads(response.content)["data"]

        ndb_client = NeuralDBClient(
            deployment_identifier=create_deployment_identifier(
                model_identifier=model_identifier,
                deployment_name=deployment_name,
                deployment_username=self._username,
            ),
            deployment_id=response_data["deployment_id"],
            bazaar=self,
        )
        if is_async:
            return ndb_client

        time.sleep(5)
        self.await_deploy(ndb_client)
        return ndb_client

    def await_deploy(self, ndb_client: NeuralDBClient):
        """
        Waits for the deployment of a model to complete.

        Args:
            ndb_client (NeuralDBClient): The NeuralDBClient instance.
        """
        url = urljoin(self._base_url, f"jobs/deploy-status")

        params = {"deployment_identifier": ndb_client.deployment_identifier}
        while True:
            response = http_get_with_error(
                url, params=params, headers=auth_header(self._access_token)
            )
            response_data = json.loads(response.content)["data"]

            if response_data["status"] == "complete":
                print("\nDeployment completed")
                return

            print("Deployment: In progress", end="", flush=True)
            print_progress_dots(duration=5)

    def undeploy(self, ndb_client: NeuralDBClient):
        """
        Undeploys a deployed model.

        Args:
            ndb_client (NeuralDBClient): The NeuralDBClient instance.
        """
        url = urljoin(self._base_url, f"jobs/undeploy")
        params = {
            "deployment_identifier": ndb_client.deployment_identifier,
        }
        response = http_post_with_error(
            url, params=params, headers=auth_header(self._access_token)
        )

        print("Deployment is shutting down.")

    def list_deployments(self):
        """
        Lists the deployments in the Model Bazaar.

        Returns:
            List[dict]: A list of dictionaries containing information about deployments.
        """
        url = urljoin(self._base_url, f"jobs/list-deployments")
        response = http_get_with_error(
            url,
            headers=auth_header(self._access_token),
        )

        response_data = json.loads(response.content)["data"]
        deployments = []
        for deployment in response_data:
            model_identifier = create_model_identifier(
                model_name=deployment["model_name"],
                author_username=deployment["model_username"],
            )
            deployment_info = {
                "deployment_identifier": create_deployment_identifier(
                    model_identifier=model_identifier,
                    deployment_name=deployment["name"],
                    deployment_username=deployment["deployment_username"],
                ),
                "status": deployment["status"],
            }
            deployments.append(deployment_info)

        return deployments

    def connect(self, deployment_identifier: str):
        """
        Connects to a deployed model and returns a NeuralDBClient instance.

        Args:
            deployment_identifier (str): The identifier of the deployment.

        Returns:
            NeuralDBClient: A NeuralDBClient instance.
        """
        url = urljoin(self._base_url, f"jobs/deploy-status")

        response = http_get_with_error(
            url,
            params={"deployment_identifier": deployment_identifier},
            headers=auth_header(self._access_token),
        )

        response_data = json.loads(response.content)["data"]

        if response_data["status"] == "complete":
            print("Connection obtained...")
            return NeuralDBClient(
                deployment_identifier=deployment_identifier,
                deployment_id=response_data["deployment_id"],
                bazaar=self,
            )

        raise Exception("The model isn't deployed...")

    def update_model(self, model_name: str, base_model_identifier: str):
        """
        Creates a new model with give name by updating the existing model with RLHF Logs.

        Args:
            model_name (str): Name for the new model.
            base_model_identifier (str): The identifier of the base model.

        Returns:
            Model: A Model instance.
        """
        url = urljoin(self._base_url, f"bazaar/rlhf-update-model")
        response = http_post_with_error(
            url,
            params={
                "model_identifier": base_model_identifier,
                "model_name": model_name,
            },
            headers=auth_header(self._access_token),
        )

        response_content = json.loads(response.content)

        if response_content["status"] != "success":
            raise Exception(response_content["message"])

        model = Model(
            model_identifier=create_model_identifier(
                model_name=model_name, author_username=self._username
            ),
            model_id=response_content["data"]["model_id"],
        )

        return model
