import thirdai
import time

# thirdai.start_heartbeat("http://localhost:8080")


# time.sleep(1)

from thirdai import bolt

print(time.time())
bolt.UniversalDeepTransformer(
    data_types={"col": bolt.types.categorical()}, target="col", n_target_classes=1
)
print(time.time())

