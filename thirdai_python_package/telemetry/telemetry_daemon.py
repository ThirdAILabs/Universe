import argparse
import signal
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

# This daemon gets run whenever a write_dir (which can be local or cloud (for
# now just s3)) is specified when starting metrics. It reads from the local
# port where the thirdai prometheus metrics are hosted, and writes those metrics
# to write_dir/telemetry-<uuid> every DEFAULT_UPLOAD_INTERVAL_SECONDS.


# See https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


# We will respond to interrupt signals (via the GracefulKiller) at this interval
DEFAULT_SLEEP_INTERVAL_SECONDS = 0.1


def push_to_local_file(parsed_file_path, raw_telemetry):
    Path(parsed_file_path.path).parent.mkdir(parents=True, exist_ok=True)
    with open(parsed_file_path.path, "wb") as f:
        f.write(raw_telemetry)


def push_to_s3(parsed_s3_path, raw_telemetry, optional_endpoint_url):
    import boto3

    if optional_endpoint_url is None:
        client = boto3.client("s3")
    else:
        client = boto3.client("s3", endpoint_url=optional_endpoint_url)
    key = parsed_s3_path.path
    if key.startswith("/"):
        key = key[1:]
    client.put_object(
        Bucket=parsed_s3_path.netloc,
        Key=key,
        Body=raw_telemetry,
    )


def parse_uuid(raw_telemetry):
    telemetry_string = raw_telemetry.decode("utf-8")
    key = "thirdai_instance_uuid"
    uuid_key_offset = telemetry_string.index(key)
    # The format of the key value pair is key="<UUID>", so we need to add the
    # length of the key + 2 to go from the offset of the key to the offset of
    # the uuid
    uuid_value_offset = uuid_key_offset + len(key) + 2
    uuid_length = 32  # 32 hex chars in a UUID
    uuid = telemetry_string[uuid_value_offset : uuid_value_offset + uuid_length]
    return uuid


def push_telemetry(push_dir, telemetry_url, optional_endpoint_url):
    raw_telemetry = requests.get(telemetry_url).content
    # Parsing the UUID from the raw telemetry instead of having it passed in
    # at program creation ensures that we are much less likely to push a
    # corrupted or wrong telemetry file to the remote location. If the parent
    # process is dead, we won't get any telemetry, so either the raw_telemetry
    # call above will fail or it will be an empty string and the parse_uuid
    # call will fail. If the parent process has died and a new process has
    # started in the meantime, we might have missed some of the parent processes
    # final updates, but we won't overwrite the parent's telemetry file with a
    # new telemetry file because the parsed uuid will be different.
    uuid = parse_uuid(raw_telemetry)
    parsed_push_location = urlparse(push_dir + "/telemetry-" + uuid)
    if parsed_push_location.scheme == "":
        push_to_local_file(parsed_push_location, raw_telemetry)
    elif parsed_push_location.scheme == "s3":
        push_to_s3(parsed_push_location, raw_telemetry, optional_endpoint_url)
    else:
        raise ValueError(f"Unknown location {push_dir}")


def launch_daemon(
    push_dir, telemetry_url, optional_endpoint_url, upload_interval_seconds, killer
):
    last_update_time = 0
    while not killer.kill_now:
        if time.time() - last_update_time > upload_interval_seconds:
            push_telemetry(push_dir, telemetry_url, optional_endpoint_url)
            last_update_time = time.time()
        # Sleeping for this shorter amount of time instead of
        # DEFAULT_UPLOAD_INTERVAL_SECONDS ensures we can respond to interrupts
        # quickly
        time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)

    # We push at the end to make sure the telemetry is flushed (if the parent
    # thirdai process has been killed this will just throw an error)
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
        "--upload_interval_seconds",
        help="How often to upload telemetry.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--optional_endpoint_url",
        help="Optional endpoint url to pass to boto3. Usually not needed (currently used for testing).",
        default=None,
    )
    args = parser.parse_args()

    killer = GracefulKiller()

    launch_daemon(
        args.push_dir,
        args.telemetry_url,
        args.optional_endpoint_url,
        args.upload_interval_seconds,
        killer,
    )
