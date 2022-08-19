import thirdai.distributed_bolt as db
import pandas as pd

TEST_FILE = "/share/pratik/test.csv"


if __name__ == "__main__":
    df = pd.read_csv(TEST_FILE)
    n_classes = df[df.columns[-1]].nunique()
    column_datatypes = []
    for col_type in df.dtypes[:-1]:
        if col_type == "int64":
            column_datatypes.append("numeric")
        elif col_type == "object":
            column_datatypes.append("categorical")
    column_datatypes.append("label")


    config_filename = "./default_config_tabular_classifier.txt"
    head = db.TabularClassifier(
        no_of_workers=2,
        config_filename=config_filename,
        num_cpus_per_node=20,
        column_datatypes=column_datatypes,
    )
    head.train(circular=False)
    acc = head.predict()
    print(acc)
