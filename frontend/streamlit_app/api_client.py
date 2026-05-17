import os
from typing import Any, Dict

import requests


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def get(path: str) -> Dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()
