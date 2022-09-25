import argparse


def decide_datasets_to_run():
    parser = argparse.ArgumentParser(
        description="Experiment with the Tabular Auto-Classifier"
    )
    parser.add_argument(
        "--run_heavy_tests",
        action="store_true",
        default=False,
        help="Includes large datasets in the experiment (several hours long). Otherwise runs on small datasets only (a few minutes only).",
    )
    args = parser.parse_args()

    datasets = [
        "ChurnModeling",
        "CensusIncome",
        "EyeMovements",
        "PokerHandInduction",
    ]
    large_datasets = [
        "OttoGroupProductClassificationChallenge",
        "BNPParibasCardifClaimsManagement",
        "ForestCoverType",
        # "HiggsBoson", # this one takes fairly long so test at your own discretion
    ]

    if args.run_heavy_tests:
        datasets += large_datasets

    return datasets


def get_col_datatypes(dataset_base_filename):
    dtypes_file = dataset_base_filename + "/Dtypes.txt"
    with open(dtypes_file) as file:
        lines = file.readlines()
        dtypes = lines[0].split(",")
        dtypes[-1] = dtypes[-1].strip("\n")
        return dtypes
