import thirdai.distributed_bolt as db
import pandas as pd

TEST_FILE = '/share/pratik/wesad_test.csv'



if __name__ == "__main__":
    df = pd.read_csv(TEST_FILE)
    n_classes = df[df.columns[-1]].nunique()
    column_datatypes = []
    for col_type in df.dtypes[:-1]:
        column_datatypes.append("numeric")
    column_datatypes.append("label")

    config_filename = "./default_config_tabular_classifier.txt"
    head = db.TabularClassifier(
        no_of_workers=4,
        config_filename=config_filename,
        num_cpus_per_node=48,
        column_datatypes=column_datatypes,
        n_classes = n_classes
    )
    head.train(circular=True)
    acc = head.predict()
    print(acc)
