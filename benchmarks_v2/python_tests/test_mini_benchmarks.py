import os

import pytest

from benchmarks_v2.src.main import main

pytestmark = [pytest.mark.unit]


def test_mini_benchmarks():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    main(
        runner=[
            "mini_benchmark_udt",
            "mini_benchmark_query_reformulation",
            "mini_benchmark_temporal",
        ],
        path_prefix=os.path.join(curr_path, "../src/mini_benchmark_datasets/"),
        fail_on_error=True,
        config="",
        mlflow_uri="",
        run_name="",
        official_slack_webhook="",
        branch_slack_webhook="",
        branch_name="",
    )
