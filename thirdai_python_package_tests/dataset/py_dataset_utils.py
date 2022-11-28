import pandas as pd


def create_csv_and_pqt_files(csv_filename, parquet_filename, csv_string):
    with open(csv_filename, "w") as file:
        file.write(content)
    df = pd.read_csv(csv_filename)
    df.to_parquet(parquet_filename)
