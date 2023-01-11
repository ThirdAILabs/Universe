import thirdai
from licensing_utils import LOCAL_HEARTBEAT_SERVER, this_should_require_a_license_bolt

if __name__ == "__main__":
    thirdai.licensing.start_heartbeat(LOCAL_HEARTBEAT_SERVER)
    this_should_require_a_license_bolt()
    thirdai.licensing.end_heartbeat()
