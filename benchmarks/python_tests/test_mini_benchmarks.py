import os

import pytest

pytestmark = [pytest.mark.unit]


def test_mini_benchmarks():
    from benchmarks.src.main import main

    curr_path = os.path.dirname(os.path.abspath(__file__))
    main(
        runner=[
            "mini_benchmark_udt",
            "mini_benchmark_query_reformulation",
            "mini_benchmark_temporal",
            "mini_benchmark_ndb",
        ],
        path_prefix=os.path.join(curr_path, "../src/mini_benchmark_datasets/"),
        config="",
        mlflow_uri="",
        run_name="",
        official_slack_webhook="",
        branch_slack_webhook="",
        branch_name="",
    )
