import thirdai
from licensing_utils import (
    LOCAL_HEARTBEAT_SERVER,
    this_should_require_a_full_license_udt,
)

if __name__ == "__main__":
    thirdai.licensing.start_heartbeat(LOCAL_HEARTBEAT_SERVER)
    this_should_require_a_full_license_udt()
    thirdai.licensing.end_heartbeat()
