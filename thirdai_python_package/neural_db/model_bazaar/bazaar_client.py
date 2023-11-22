import json
import time
from pathlib import Path
from typing import List, Union
from urllib.parse import urljoin
from uuid import UUID

from .bazaar_base import Bazaar, auth_header
from .utils import (
    check_deployment_decorator,
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

    def __init__(self, model_identifier) -> None:
        self._model_identifier = model_identifier

    @property
    def model_identifier(self):
        return self._model_identifier


class NeuralDBClient:
    """
    A client for interacting with the deployed NeuralDB model.

    Attributes:
        deployment_identifier (str): The identifier for the deployment.
        base_url (str): The base URL for the deployed NeuralDB model.

    Methods:
        __init__(self, deployment_identifier: str, base_url: str) -> None:
            Initializes a new instance of the NeuralDBClient.

        search(self, query: str, top_k: int = 10) -> List[dict]:
            Searches the ndb model for relevant search results.

        insert(self, files: List[str]) -> None:
            Inserts documents into the ndb model.

        associate(self, query1: str, query2: str) -> None:
            Associates two queries in the ndb model.

        upvote(self, query_id: UUID, query_text: str, reference: dict) -> None:
            Upvotes a response in the ndb model.

        downvote(self, query_id: UUID, query_text: str, reference: dict) -> None:
            Downvotes a response in the ndb model.
    """

    def __init__(self, deployment_identifier, base_url):
        """
        Initializes a new instance of the NeuralDBClient.

        Args:
            deployment_identifier (str): The identifier for the deployment.
            base_url (str): The base URL for the deployed NeuralDB model.
        """
        self.deployment_identifier = deployment_identifier
        self.base_url = base_url

    @check_deployment_decorator
    def search(self, query, top_k=10):
        """
        Searches the ndb model for similar queries.

        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to retrieve (default is 10).

        Returns:
            List[dict]: A list of search results.
        """
        response = http_get_with_error(
            urljoin(self.base_url, "predict"),
            params={"query_text": query, "top_k": top_k},
        )

        return json.loads(response.content)["data"]

    @check_deployment_decorator
    def insert(self, files: List[str]):
        """
        Inserts documents into the ndb model.

        Args:
            files (List[str]): A list of file paths to be inserted into the ndb model.
        """
        files = [("files", open(file_path, "rb")) for file_path in files]
        response = http_post_with_error(urljoin(self.base_url, "insert"), files=files)

        print(json.loads(response.content)["message"])

    @check_deployment_decorator
    def associate(self, query1: str, query2: str):
        """
        Associates two queries in the ndb model.

        Args:
            query1 (str): The first query.
            query2 (str): The second query.
        """
        response = http_post_with_error(
            urljoin(self.base_url, "associate"),
            json={"query1": query1, "query2": query2},
        )

    @check_deployment_decorator
    def upvote(self, query_id: UUID, query_text: str, reference: dict):
        """
        Upvotes a response in the ndb model.

        Args:
            query_id (UUID): The ID of the query.
            query_text (str): The text of the query.
            reference (dict): A dictionary containing the reference information.
        """
        response = http_post_with_error(
            urljoin(self.base_url, "upvote"),
            json={
                "query_id": query_id,
                "query_text": query_text,
                "reference_id": reference["id"],
                "reference_text": reference["text"],
            },
        )

        print("Successfully upvoted the specified search result.")

    @check_deployment_decorator
    def downvote(self, query_id: UUID, query_text: str, reference: dict):
        """
        Downvotes a response in the ndb model.

        Args:
            query_id (UUID): The ID of the query.
            query_text (str): The text of the query.
            reference (dict): A dictionary containing the reference information.
        """
        response = http_post_with_error(
            urljoin(self.base_url, "downvote"),
            json={
                "query_id": query_id,
                "query_text": query_text,
                "reference_id": reference["id"],
                "reference_text": reference["text"],
            },
        )

        print("Successfully downvoted the specified search result.")


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

        push_model(self, model_name: str, local_path: str, access_level: str = "public") -> None:
            Pushes a model to the Model Bazaar.

        pull_model(self, model_identifier: str) -> NeuralDBClient:
            Pulls a model from the Model Bazaar and returns a NeuralDBClient instance.

        list_models(self) -> List[dict]:
            Lists available models in the Model Bazaar.

        train(self, model_name: str, docs: List[str], is_async: bool = False, base_model_identifier: str = None) -> Model:
            Initiates training for a model and returns a Model instance.

        await_train(self, model: Model) -> None:
            Waits for the training of a model to complete.

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
        self._user_id = None
        self._access_token = None

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
        self._user_id = self._login_instance.user_id
        self._access_token = self._login_instance.access_token
        self._username = self._login_instance._username

    def push_model(
        self, model_name: str, local_path: str, access_level: str = "public"
    ):
        """
        Pushes a model to the Model Bazaar.

        Args:
            model_name (str): The name of the model.
            local_path (str): The local path of the model.
            access_level (str): The access level for the model (default is "public").
        """
        self.push(
            name=model_name,
            model_path=local_path,
            trained_on="Own Documents",
            access_level=access_level,
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
        docs: List[str],
        is_async: bool = False,
        base_model_identifier: str = None,
    ):
        """
        Initiates training for a model and returns a Model instance.

        Args:
            model_name (str): The name of the model.
            docs (List[str]): A list of document paths for training.
            is_async (bool): Whether training should be asynchronous (default is False).
            base_model_identifier (str): The identifier of the base model (optional).

        Returns:
            Model: A Model instance.
        """
        url = urljoin(self._base_url, f"jobs/{self._user_id}/train")
        files = [("files", open(file_path, "rb")) for file_path in docs]

        response = http_post_with_error(
            url,
            params={
                "model_name": model_name,
                "base_model_identifier": base_model_identifier,
            },
            files=files,
            headers=auth_header(self._access_token),
        )
        response_data = json.loads(response.content)["data"]
        model = Model(
            model_identifier=create_model_identifier(
                model_name=model_name, author_username=self._username
            ),
        )

        if is_async:
            return model

        self.await_train(model)
        return model

    def await_train(self, model: Model):
        """
        Waits for the training of a model to complete.

        Args:
            model (Model): The Model instance.
        """
        url = urljoin(self._base_url, f"jobs/{self._user_id}/train-status")
        while True:
            response = http_get_with_error(
                url,
                params={"model_identifier": model.model_identifier},
                headers=auth_header(self._access_token),
            )
            response_data = json.loads(response.content)["data"]

            if response_data["status"] == "complete":
                print("\nTraining completed")
                return

            print("Training in progress", end="", flush=True)
            print_progress_dots(duration=5)

    def deploy(self, model_identifier: str, deployment_name: str, is_async=False):
        """
        Deploys a model and returns a NeuralDBClient instance.

        Args:
            model_identifier (str): The identifier of the model.
            deployment_name (str): The name for the deployment.
            is_async (bool): Whether deployment should be asynchronous (default is False).

        Returns:
            NeuralDBClient: A NeuralDBClient instance.
        """
        url = urljoin(self._base_url, f"jobs/{self._user_id}/deploy")
        params = {
            "user_id": self._user_id,
            "model_identifier": model_identifier,
            "deployment_name": deployment_name,
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
            base_url=response_data["endpoint"] + "/",
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
        url = urljoin(self._base_url, f"jobs/{self._user_id}/deploy-status")

        params = {"deployment_identifier": ndb_client.deployment_identifier}
        while True:
            response = http_get_with_error(
                url, params=params, headers=auth_header(self._access_token)
            )
            response_data = json.loads(response.content)["data"]

            if response_data["status"] == "complete":
                print("\nDeployment completed")
                return

            print("Deployment in progress", end="", flush=True)
            print_progress_dots(duration=5)

    def undeploy(self, ndb_client: NeuralDBClient):
        """
        Undeploys a deployed model.

        Args:
            ndb_client (NeuralDBClient): The NeuralDBClient instance.
        """
        url = urljoin(self._base_url, f"jobs/{self._user_id}/undeploy")
        params = {
            "deployment_identifier": ndb_client.deployment_identifier,
        }
        response = http_post_with_error(
            url, params=params, headers=auth_header(self._access_token)
        )

        print(response)

        print("Deployment is shutting down.")

    def list_deployments(self):
        """
        Lists the deployments in the Model Bazaar.

        Returns:
            List[dict]: A list of dictionaries containing information about deployments.
        """
        url = urljoin(self._base_url, f"jobs/{self._user_id}/list-deployments")
        response = http_get_with_error(
            url,
            params={
                "user_id": self._user_id,
            },
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
        url = urljoin(self._base_url, f"jobs/{self._user_id}/deploy-status")

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
                base_url=response_data["endpoint"] + "/",
            )

        raise Exception("The model isn't deployed...")
