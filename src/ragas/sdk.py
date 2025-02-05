"""
SDK module for interacting with the Ragas API service.
"""
import json
import os
from functools import lru_cache

import requests

from ragas._analytics import get_userid
from ragas._version import __version__
from ragas.exceptions import UploadException
from ragas.utils import base_logger
from datetime import datetime, timezone

# endpoint for uploading results
RAGAS_API_URL = "https://api.ragas.io"
RAGAS_APP_URL = "https://app.ragas.io"
RAGAS_API_SOURCE = "ragas_py"


@lru_cache(maxsize=1)
def get_app_token() -> str:
    app_token = os.environ.get("RAGAS_APP_TOKEN")
    if app_token is None:
        raise ValueError("RAGAS_APP_TOKEN is not set")
    return app_token


@lru_cache(maxsize=1)
def get_api_url() -> str:
    return os.environ.get("RAGAS_API_URL", RAGAS_API_URL)


@lru_cache(maxsize=1)
def get_app_url() -> str:
    return os.environ.get("RAGAS_APP_URL", RAGAS_APP_URL)


@lru_cache(maxsize=1)
def get_enable_http_log() -> bool:
    log_value = os.environ.get("RAGAS_ENABLE_HTTP_LOG", "false").lower()
    return log_value == "true"


def upload_packet(path: str, data_json_string: str):
    app_token = get_app_token()
    base_url = get_api_url()
    app_url = get_app_url()

    connection_timeout = 300  # 5 minutes
    read_timeout = 300  # 5 minutes

    headers = {
        "Content-Type": "application/json",
        "x-app-token": app_token,
        "x-source": RAGAS_API_SOURCE,
        "x-app-version": __version__,
        "x-ragas-lib-user-uuid": get_userid(),
    }

    delimiter = "=" * 80
    section_delimiter = "-" * 30
    api_url = f"{base_url}/api/v1{path}"

    enable_http_log = get_enable_http_log()
    if enable_http_log:
        start_time = datetime.now(timezone.utc)
        print(f"\n\n{delimiter}")
        print(f"Logging started at: {start_time}")
        print(section_delimiter)

        print(f"api_url: {api_url}")
        print(section_delimiter)

        print(f"base_url: {base_url}")
        print(section_delimiter)

        print(f"app_url: {app_url}")
        print(section_delimiter)

        print("timeout_config:")
        print(f"  connection_timeout: {connection_timeout}s")
        print(f"  read_timeout: {read_timeout}s")
        print(section_delimiter)

        # Create a copy of headers and set x-app-token to [REDACTED] if it exists
        log_headers = headers.copy()
        if "x-app-token" in log_headers:
            log_headers["x-app-token"] = "***[REDACTED]***"

        print("\nheaders:")
        for key, value in log_headers.items():
            print(f"  {key}: {value}")
        print(section_delimiter)

        print("\ndata_json:")
        print(f"  {data_json_string}")
        print(section_delimiter)

    response = requests.post(
        f"{base_url}/api/v1{path}",
        data=data_json_string,
        headers=headers,
        timeout=(connection_timeout, read_timeout)
    )

    if enable_http_log:
        try:
            response_data = response.json()
            print("\nresponse:")
            if response.status_code >= 400:
                print("  status: ERROR")
            else:
                print("  status: SUCCESS")
            print(f"  status_code: {response.status_code}")
            print("  data:")
            print(f"    {json.dumps(response_data, indent=2)}")
        except Exception:
            print("\nresponse:")
            print("  status: ERROR" if response.status_code >= 400 else "  status: SUCCESS")
            print(f"  status_code: {response.status_code}")
            print("  data:")
            print(f"    {response.text}")
        print(section_delimiter)

        print("Logging ended")
        print(f"{delimiter}\n\n")

    check_api_response(response)
    return response


def check_api_response(response: requests.Response) -> None:
    """
    Check API response status and raise appropriate exceptions

    Parameters
    ----------
    response : requests.Response
        Response object from API request

    Raises
    ------
    UploadException
        If authentication fails or other API errors occur
    """
    if response.status_code == 403:
        base_logger.error(
            "[AUTHENTICATION_ERROR] The app token is invalid. "
            "Please check your RAGAS_APP_TOKEN environment variable. "
            f"Response Status: {response.status_code}, URL: {response.url}"
        )
        raise UploadException(
            status_code=response.status_code,
            message="AUTHENTICATION_ERROR: The app token is invalid. Please check your RAGAS_APP_TOKEN environment variable.",
        )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        error_msg = ""
        try:
            error_data = response.json()
            if "message" in error_data:
                error_msg += f"\nAPI Message: {error_data['message']}"
            if "debug_error_info" in error_data:
                error_msg += f"\nDebug Info: {error_data['debug_error_info']}"
        except Exception as _e:
            error_msg = f"\nStatus Code: {response.status_code}"

        base_logger.error(
            f"[API_ERROR] Request failed. "
            f"Status Code: {response.status_code}, URL: {response.url}, "
            f"Error Message: {error_msg}"
        )
        raise UploadException(
            status_code=response.status_code, message=f"Request failed: {error_msg}"
        )


def build_evaluation_app_url(app_url: str, run_id: str) -> str:
    return f"{app_url}/dashboard/alignment/evaluation/{run_id}"
