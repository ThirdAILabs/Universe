import time

import pandas as pd
from thirdai import bolt, demos, telemetry

telemetry.start(write_dir="s3://thirdai-corp/metrics/demo", write_period_seconds=5)

model = bolt.UniversalDeepTransformer(
    data_types={
        "category": bolt.types.categorical(),
        "text": bolt.types.text(),
    },
    target="category",
    n_target_classes=150,
    integer_target=True,
)

train_filename, test_filename, _ = demos.download_clinc_dataset()

model.train(train_filename)

test_file = pd.read_csv(test_filename)["text"]

for t in test_file:
    model.predict({"text": t}, return_predicted_class=True)
    time.sleep(0.1)
    model.explain({"text": t})
