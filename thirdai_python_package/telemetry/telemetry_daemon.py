import argparse
import signal
import time
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
    with open(parsed_file_path.path, "w") as f:
        f.write(raw_telemetry)


def s3_daemon(parsed_s3_path, raw_telemetry):
    import boto3

    s3 = boto3.resource("s3")
    s3.put_object(
        Bucket=parsed_s3_path.netloc,
        Key=parsed_s3_path.path,
        Body=raw_telemetry,
    )
    pass


def push_telemetry(push_location, telemetry_url):
    raw_telemetry = requests.get(telemetry_url).content
    parsed_push_location = urlparse(push_location)
    if parsed_push_location.scheme == "":
        local_file_daemon(parsed_push_location, telemetry_url)
    elif parsed_push_location.scheme == "s3":
        s3_daemon(parsed_push_location, raw_telemetry)
    else:
        raise ValueError(f"Unknown location {push_location}")


def launch_daemon(push_location, telemetry_url, killer):
    last_update_time = 0
    while not killer.kill_now:
        if time.time() - last_update_time > DEFAULT_UPLOAD_INTERVAL:
            push_telemetry(push_location, telemetry_url)
            last_update_time = time.time()
        time.sleep(DEFAULT_SLEEP_INTERVAL)

    push_telemetry(push_location, telemetry_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start a background daemon thread that pushes telemetry to a remote location."
    )
    parser.add_argument(
        "--telemetry_url", help="The local telemetry server url to scrape from."
    )
    parser.add_argument(
        "--push_location",
        help="The location (currently local or s3) to push telemetry to.",
    )
    args = parser.parse_args()

    killer = GracefulKiller()

    launch_daemon(args.push_location, args.telemetry_url, killer)
