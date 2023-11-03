import os
from functools import lru_cache

import yaml
from sqlalchemy import create_engine
from thirdai import neural_db as ndb

CONNECTOR_DOCS_BASE_DIR = os.path.dirname(os.path.dirname(__file__))


@lru_cache
def get_creds():
    settings = dict()
    settings_path = os.path.join(
        CONNECTOR_DOCS_BASE_DIR, "connectors_object", "creds.yaml"
    )
    if os.path.isfile(settings_path):
        with open(settings_path, mode="r") as f:
            settings.update(yaml.load(f, Loader=yaml.FullLoader))
    return settings


def get_sql_engine():
    db_url = "sqlite:///" + os.path.join(
        CONNECTOR_DOCS_BASE_DIR, "SQL", "Amazon_polarity.db"
    )
    return create_engine(db_url)


def get_sql_table():
    creds = get_creds()
    return creds["sqlite"]["table_name"]


def get_client_context():
    creds = get_creds()
    sp_creds = creds["sharePoint"]
    return ndb.SharePoint.setup_clientContext(
        base_url=sp_creds["site_url"], credentials=sp_creds
    )


def get_library_path():
    creds = get_creds()
    return creds["sharePoint"]["library_path"]


def get_base_connectors(doc):
    if isinstance(doc, ndb.SQLDatabase):
        return get_sql_engine()
    elif isinstance(doc, ndb.SharePoint):
        return get_client_context()

    raise TypeError("Unsupported document connector type")
