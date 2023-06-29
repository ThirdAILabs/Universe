from argparse import ArgumentParser
from thirdai import bolt, bolt_v2
import thirdai
import pandas as pd
# thirdai.licensing.set_path("/share/data/Shared with ThirdAI/license.serialized")

thirdai.logging.setup(log_to_stderr = False, path = "train_log_v3")

file = "clinc_train.csv"
df = pd.read_csv(file)
print(df.head())


targets = df["category"].nunique()
model = bolt.UniversalDeepTransformer(
    data_types = {
        "text": bolt.types.text(),
        "category": bolt.types.categorical(),
    },
    target="category",
    n_target_classes=targets,
    integer_target=True,
)

lr = 0.005

# callback = bolt_v2.train.callbacks.Overfitting("train_categorical_accuracy", 0.9)

# metrics = model.cold_start(
#     filename=file,
#     strong_column_names=["text"],
#     weak_column_names=[],
#     learning_rate=lr,
#     metrics=["precision@1"],
#     max_in_memory_batches = 1,
#     epochs=2
#     # callbacks = [callback],
# )

# model.train(filename=file,epochs=3,learning_rate=lr,batch_size=5, max_in_memory_batches=7, metrics=["categorical_accuracy"])
model.train(filename=file,epochs=5,learning_rate=lr,batch_size=5, metrics=["categorical_accuracy"])
# model.save(f"/share/data/Shared with ThirdAI/IPR-{folder}/para_mach_10_epoch.bolt")
print(model.predict_batch([{"text":"I am an Engineer"}, {"text":"Hey there fdskjfd fdsjfsdkjns fsdjfskjfn sfdjnfksjn"}], top_k=5, return_predicted_class = False))
# print(model.predict_batch([{"text":"I am an Engineer"}, {"text":"Hey there"}], return_predicted_class = False))
# print(model.predict({"text":"I am an Engineer"}, top_k = 5,return_predicted_class = False))

# print(metrics)
