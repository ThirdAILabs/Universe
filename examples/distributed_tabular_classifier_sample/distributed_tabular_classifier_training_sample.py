import thirdai.distributed_bolt as db
import pandas as pd

dataset_base_filename = '/share/data/tabular_benchmarks/HiggsBoson'

TEST_FILE = dataset_base_filename + '/Test.csv'


def get_col_datatypes(dataset_base_filename):
    dtypes_file = dataset_base_filename + "/Dtypes.txt"
    with open(dtypes_file) as file:
        lines = file.readlines()
        dtypes = lines[0].split(",")
        dtypes[-1] = dtypes[-1].strip("\n")
        return dtypes

if __name__ == "__main__":
    df = pd.read_csv(TEST_FILE)
    n_classes = df[df.columns[-1]].nunique()
    # column_datatypes = []
    # for col_type in df.dtypes[:-1]:
    #     column_datatypes.append("numeric")
    # column_datatypes.append("label")

    config_filename = "./default_config_tabular_classifier.txt"
    head = db.TabularClassifier(
        no_of_workers=2,
        config_filename=config_filename,
        num_cpus_per_node=20,
        column_datatypes=get_col_datatypes(dataset_base_filename),
        n_classes = n_classes
    )
    head.train(circular=True)
    acc = head.predict()
    print(acc)
