from pathlib import Path
from typing import Union, List
import json
from urllib.parse import urljoin
import requests
import time
from uuid import UUID
from functools import wraps

from bazaar import Bazaar, auth_header
from utils import (
    http_get_with_error,
    http_post_with_error,
    create_model_identifier,
    create_deployment_identifier,
    print_progress_dots,
)


class Model:
    def __init__(self, user_id, model_id, model_identifier=None) -> None:
        self._user_id = user_id
        self._model_id = model_id
        self._model_identifier = model_identifier

    @property
    def user_id(self):
        return self._user_id

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_identifier(self):
        return self._model_identifier


def check_deployment_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.base_url is None:
            raise Exception(
                "Deployment isn't complete yet. Use `list_deployment()` to check status."
            )
        return func(self, *args, **kwargs)

    return wrapper


class NeuralDBClient:
    def __init__(self, deployment_identifier, base_url=None):
        self.deployment_identifier = deployment_identifier
        self.base_url = base_url

    @check_deployment_decorator
    def search(self, query, top_k=10):
        # Query the ndb model
        response = http_get_with_error(
            urljoin(self.base_url, "predict"),
            params={"query_text": query, "top_k": top_k},
        )

        return json.loads(response.content)["data"]

    @check_deployment_decorator
    def insert(self, files: List[str]):
        # Upload document to index into ndb model
        files = [("files", open(file_path, "rb")) for file_path in files]
        response = http_post_with_error(urljoin(self.base_url, "insert"), files=files)

        print(json.loads(response.content)["message"])

    @check_deployment_decorator
    def associate(self, query1: str, query2: str):
        # Associate two queries
        response = http_post_with_error(
            urljoin(self.base_url, "associate"),
            json={"query1": query1, "query2": query2},
        )

    @check_deployment_decorator
    def upvote(self, query_id: UUID, query_text: str, reference: dict):
        # Upvote a response
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
        # Downvote a response
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
    def __init__(
        self,
        base_url: str,
        cache_dir: Union[Path, str],
    ):
        super().__init__(cache_dir, base_url)
        self._username = None
        self._user_id = None
        self._access_token = None

    def sign_up(self, email, password, username):
        self.signup(email=email, password=password, username=username)
        self._username = username

    def log_in(self, email, password):
        self.login(email=email, password=password)
        self._user_id = self._login_instance.user_id
        self._access_token = self._login_instance.access_token
        self._username = self._login_instance._username

    def push_model(
        self, model_name: str, local_path: str, access_level: str = "public"
    ):
        self.push(
            name=model_name,
            model_path=local_path,
            trained_on="Own Documents",
            access_level=access_level,
        )

    def pull_model(self, model_identifier: str):
        return self.get_neuraldb(model_identifier=model_identifier)

    def list_models(self):
        return self.fetch()

    def train(
        self,
        model_name: str,
        docs: List[str],
        is_async: bool = False,
        base_model_identifier: str = None,
    ):
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
            user_id=response_data["user_id"],
            model_id=response_data["model_id"],
            model_identifier=create_model_identifier(
                model_name=model_name, author_username=self._username
            ),
        )

        if is_async:
            return model

        self.await_train(model)
        return model

    def await_train(self, model: Model):
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
        url = urljoin(self._base_url, f"jobs/{self._user_id}/deploy")
        params = {
            "user_id": self._user_id,
            "model_identifier": model_identifier,
            "deployment_name": deployment_name,
        }
        response = http_post_with_error(
            url, params=params, headers=auth_header(self._access_token)
        )

        ndb_client = NeuralDBClient(
            deployment_identifier=create_deployment_identifier(
                model_identifier=model_identifier,
                deployment_name=deployment_name,
                deployment_username=self._username,
            ),
            base_url=None,
        )
        if is_async:
            return ndb_client

        time.sleep(5)
        self.await_deploy(ndb_client)
        return ndb_client

    def await_deploy(self, ndb_client: NeuralDBClient):
        url = urljoin(self._base_url, f"jobs/{self._user_id}/deploy-status")

        params = {"deployment_identifier": ndb_client.deployment_identifier}
        while True:
            response = http_get_with_error(
                url, params=params, headers=auth_header(self._access_token)
            )
            response_data = json.loads(response.content)["data"]

            if response_data["status"] == "complete":
                print("\nDeployment completed")
                ndb_client.base_url = response_data["endpoint"] + "/"
                return

            print("Deployment in progress", end="", flush=True)
            print_progress_dots(duration=5)

    def undeploy(self, ndb_client: NeuralDBClient):
        url = urljoin(self._base_url, f"jobs/{self._user_id}/undeploy")
        params = {
            "deployment_identifier": ndb_client.deployment_identifier,
        }
        response = http_post_with_error(
            url, params=params, headers=auth_header(self._access_token)
        )

        print("Deployment is shutting down.")

    def list_deployments(self):
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
                "deployemnt_identifier": create_deployment_identifier(
                    model_identifier=model_identifier,
                    deployment_name=deployment["name"],
                    deployment_username=deployment["deployment_username"],
                ),
                "status": deployment["status"],
            }
            deployments.append(deployment_info)

        return deployments

    def connect(self, deployment_identifier: str):
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
