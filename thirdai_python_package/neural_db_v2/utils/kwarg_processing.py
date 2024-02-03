from typing import Any, Dict


def extract_kwargs(kwargs: Dict[str, Any], prefix: str):
    return {
        key[len(prefix) :]: value
        for key, value in kwargs.items()
        if key.startswith(prefix)
    }
