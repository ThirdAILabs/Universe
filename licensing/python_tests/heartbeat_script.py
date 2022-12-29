import thirdai
from licensing_utils import LOCAL_HEARTBEAT_SERVER, this_should_require_a_license_bolt

if __name__ == "__main__":
    thirdai.start_heartbeat(LOCAL_HEARTBEAT_SERVER)
    this_should_require_a_license_bolt()
    thirdai.end_heartbeat()
