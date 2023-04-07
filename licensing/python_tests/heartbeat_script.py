import thirdai
from licensing_utils import LOCAL_HEARTBEAT_SERVER, run_udt_training_routine

if __name__ == "__main__":
    thirdai.licensing.start_heartbeat(LOCAL_HEARTBEAT_SERVER)
    run_udt_training_routine()
    thirdai.licensing.end_heartbeat()
