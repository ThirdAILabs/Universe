from __future__ import annotations
import pandas as pd
from openai import OpenAI

from .templates.model_templates import (
    TabularClassificationTemplate,
    RegressionTemplate,
    TokenClassificationTemplate,
    QueryReformulationTemplate,
    supported_templates,
)

from .column_inferencing import column_detector
import warnings

warnings.filterwarnings("ignore")


def raise_exception_without_trace(message):
    raise Exception(message) from None


def verify_dataframe(dataframe: pd.DataFrame, target_column_name: str, task: str):
    if target_column_name not in dataframe.columns:
        raise Exception("Specified target column not found in the dataframe")

    if len(dataframe) < 50 and task is None:
        raise Exception(
            f"Minimum required rows to infer the problem type is 50 but dataframe has number rows {len(dataframe)}"
        )

    if len(dataframe) == 0:
        raise Exception(f"Cannot detect a task for dataset with 0 rows.")


def auto_infer_model_builder(target_column_name: str, dataframe: pd.DataFrame):
    # approx representation of a column
    target_column = column_detector.detect_single_column_type(
        target_column_name, dataframe
    )

    input_columns = column_detector.get_input_columns(target_column_name, dataframe)

    if isinstance(target_column, column_detector.NumericalColumn):
        return RegressionTemplate(
            dataframe,
            target_column,
            input_columns,
        )

    if isinstance(target_column, column_detector.CategoricalColumn):

        if target_column.number_tokens_per_row >= 4:

            token_column_candidates = (
                column_detector.get_token_candidates_for_token_classification(
                    target_column, input_columns
                )
            )

            if (
                len(token_column_candidates) == 1
                and target_column.unique_tokens_per_row * len(dataframe) < 250
            ):
                return TokenClassificationTemplate(
                    dataframe,
                    target_column,
                    input_columns,
                )

            source_column_candidates = (
                column_detector.get_source_column_for_query_reformulation(
                    target_column, input_columns
                )
            )

            if len(source_column_candidates) == 1 and target_column.token_type == "str":
                return QueryReformulationTemplate(
                    dataframe,
                    target_column,
                    input_columns,
                )

        return TabularClassificationTemplate(
            dataframe,
            target_column,
            input_columns,
        )

    raise Exception(
        "Could not automatically infer task using the provided column name and the template. The following target types are supported for classification : Numerical, Categorical, and Text. Verify that the target column has one of the following types or explicitly specify the task. Check out https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer to learn more about how to initialize and train a UniversalDeepTransformer."
    )


def query_gpt(prompt, model_name, client):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def get_task_detection_prompt(query: str):
    prompt = "I have 6 different task types. Here is the description of each of the task :- \n"
    for task_id, task_template in enumerate(supported_templates):
        prompt += f"{task_id} : Task : {task_template.task}, Description: {task_template.description}, Keywords : {' '.join(task_template.keywords)}\n"

    prompt += (
        "Which task amongst the above is the closest to the following problem : \n"
        + query
    )
    prompt += (
        "\nonly return the task number and nothing else (this is extremely important)."
    )

    return prompt


def get_task_template_from_query(query: str, openai_client: OpenAI):
    prompt = get_task_detection_prompt(query)

    response = query_gpt(prompt, model_name="gpt-4", client=openai_client)
    response = "".join([char for char in response if char.isdigit()])

    try:
        template_id = int(response)
        return supported_templates[template_id]
    except:
        print("Oops ChatGPT wrong output")
        return None
