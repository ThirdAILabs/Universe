import argparse
import signal
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


# See https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


DEFAULT_SLEEP_INTERVAL = 0.1
DEFAULT_UPLOAD_INTERVAL = 60 * 20

# TODO(Check for parent process id to gracefully shut down)


def local_file_daemon(parsed_file_path, raw_telemetry):
    Path(parsed_file_path.path).parent.mkdir(parents=True, exist_ok=True)
    with open(parsed_file_path.path, "wb") as f:
        f.write(raw_telemetry)


def s3_daemon(parsed_s3_path, raw_telemetry, optional_endpoint_url):
    import boto3

    client = boto3.client("s3", endpoint_url=optional_endpoint_url)
    client.put_object(
        Bucket=parsed_s3_path.netloc,
        Key=parsed_s3_path.path,
        Body=raw_telemetry,
    )


def parse_uuid(raw_telemetry):
    telemetry_string = raw_telemetry.decode("utf-8")
    uuid_key_index = telemetry_string.index("thirdai_instance_uuid")
    uuid = telemetry_string[uuid_key_index + 23 : uuid_key_index + 23 + 32]
    return uuid


def push_telemetry(push_dir, telemetry_url, optional_endpoint_url):
    raw_telemetry = requests.get(telemetry_url).content
    uuid = parse_uuid(raw_telemetry)
    parsed_push_location = urlparse(push_dir + "/telemetry-" + uuid)
    if parsed_push_location.scheme == "":
        local_file_daemon(parsed_push_location, raw_telemetry)
    elif parsed_push_location.scheme == "s3":
        s3_daemon(parsed_push_location, raw_telemetry, optional_endpoint_url)
    else:
        raise ValueError(f"Unknown location {push_dir}")


def launch_daemon(push_dir, telemetry_url, optional_endpoint_url, killer):
    last_update_time = 0
    while not killer.kill_now:
        if time.time() - last_update_time > DEFAULT_UPLOAD_INTERVAL:
            push_telemetry(push_dir, telemetry_url, optional_endpoint_url)
            last_update_time = time.time()
        time.sleep(DEFAULT_SLEEP_INTERVAL)

    push_telemetry(push_dir, telemetry_url, optional_endpoint_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start a background daemon thread that pushes telemetry to a remote location."
    )
    parser.add_argument(
        "--telemetry_url",
        help="The local telemetry server url to scrape from.",
        required=True,
    )
    parser.add_argument(
        "--push_dir",
        help="The location (currently local or s3) to push telemetry to.",
        required=True,
    )
    parser.add_argument(
        "--optional_endpoint_url",
        help="Optional endpoint url to pass to boto3. Usually not needed (currently used for testing).",
        required=True,
    )
    args = parser.parse_args()

    killer = GracefulKiller()

    launch_daemon(args.push_dir, args.telemetry_url, args.optional_endpoint_url, killer)
