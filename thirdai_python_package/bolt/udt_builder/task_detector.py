from __future__ import annotations

import warnings

import pandas as pd
from openai import OpenAI

from .column_inferencing import column_detector
from .model_templates import (
    SUPPORTED_TASK_TYPES,
    SUPPORTED_TEMPLATES,
    TASK_TO_TEMPLATE_MAP,
    QueryReformulationTemplate,
    RecurrentClassifierTemplate,
    RegressionTemplate,
    TabularClassificationTemplate,
    TokenClassificationTemplate,
    UDTDataTemplate,
)

TEMPLATE_DETECTION_MESSAGE = """
1. Provide more information about the task
    bolt.UniversalDeepTransformer(
        dataset_path = {dataset_path},
        target_column = {target},
        task = 'more information about the task'
    )\n
2. Choose from the following list of available tasks:
{template_names}
To detect a different task, call the detect function again with:
    bolt.UniversalDeepTransformer(
        dataset_path = {dataset_path},
        target_column = {target},
        task = 'your_selected_task'
    )\n
3. Explicitly specify the datatypes.
 Check out https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer to learn how to initialize a model using explicit data types.
"""

warnings.filterwarnings("ignore")


def verify_dataframe(dataframe: pd.DataFrame, target_column_name: str, task: str):
    if target_column_name not in dataframe.columns:
        raise Exception("Specified target column not found in the dataframe")

    if len(dataframe) < 50 and task is None:
        raise Exception(
            f"Minimum required rows to infer the problem type is 50 but dataframe has number rows {len(dataframe)}"
        )

    if len(dataframe) == 0:
        raise Exception(f"Cannot detect a task for dataset with 0 rows.")


def auto_infer_task_template(target_column_name: str, dataframe: pd.DataFrame):
    """
    Analyzes the target column's data type and relationships with other columns in the dataframe to determine the most
    relevant task type among four options: regression, token classification, query reformulation, and tabular classification.

    Heuristics:
    - If the target column contains numeric data, it infers a regression task.
    - If the target column is categorical:
        - If another column matches the exact number of tokens, it infers token classification.
        - If another column has roughly the same but not exact number of tokens (tolerance is +/-30%), it infers query reformulation.
        - If no such relationships are found, it defaults to tabular classification.
    """

    # approx representation of a column
    target_column = column_detector.detect_single_column_type(
        target_column_name, dataframe
    )

    input_columns = column_detector.get_input_columns(target_column_name, dataframe)

    if isinstance(target_column, column_detector.NumericalColumn):
        return RegressionTemplate(dataframe, target_column, input_columns)

    if isinstance(target_column, column_detector.CategoricalColumn):

        if target_column.token_type == str and target_column.number_tokens_per_row >= 4:

            # check if task can be token classification
            token_column_candidates = (
                column_detector.get_token_candidates_for_token_classification(
                    target_column, input_columns
                )
            )
            if (
                len(token_column_candidates) == 1
                and target_column.unique_tokens_per_row * len(dataframe) < 250
            ):
                concrete_target_column = (
                    TokenClassificationTemplate.get_concrete_target(
                        target_column, dataframe
                    )
                )
                concrete_input_columns = {
                    token_column_candidates[0].name: column_detector.TextColumn(
                        name=token_column_candidates[0].name
                    )
                }
                return TokenClassificationTemplate(
                    dataframe, concrete_target_column, concrete_input_columns
                )

            # check if task can be query reformulation
            source_column_candidates = (
                column_detector.get_source_column_for_query_reformulation(
                    target_column, input_columns
                )
            )

            if len(source_column_candidates) == 1:
                concrete_target_column = column_detector.TextColumn(
                    name=target_column.name
                )
                concrete_input_columns = {
                    source_column_candidates[0].name: column_detector.TextColumn(
                        name=source_column_candidates[0].name
                    )
                }
                return QueryReformulationTemplate(
                    dataframe, concrete_target_column, concrete_input_columns
                )

        return TabularClassificationTemplate(dataframe, target_column, input_columns)

    raise Exception(
        "Could not automatically infer task using the provided column name and the template. The following target types are supported for classification : Numerical, Categorical, and Text. Verify that the target column has one of the following types or explicitly specify the task. Check out https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer to learn more about how to initialize and train a UniversalDeepTransformer."
    )


def query_gpt(prompt, model_name, client):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0
    )
    return response.choices[0].message.content


def get_task_detection_prompt(query: str):
    prompt = "I have 6 different task types. Here is the description of each of the task :- \n"
    for task_id, task_template in enumerate(SUPPORTED_TEMPLATES):
        prompt += f"{task_id} - Task : {task_template.task}, Description: {task_template.description}, Keywords : {' '.join(task_template.keywords)}\n".lower()

    prompt += (
        "Which task amongst the above is the closest to the following problem : \n"
        + query.lower()
    )
    prompt += "\nonly return the task number (as an int) and nothing else (this is extremely important)."

    return prompt


def get_template_from_query(
    query: str, openai_client: OpenAI, target_column: str, dataframe: pd.DataFrame
):
    detected_template = None

    if query in TASK_TO_TEMPLATE_MAP:
        detected_template = TASK_TO_TEMPLATE_MAP[query]

    if openai_client:
        prompt = get_task_detection_prompt(query)
        response = query_gpt(prompt, model_name="gpt-4", client=openai_client)
        response = "".join([char for char in response if char.isdigit()])
        try:
            template_id = int(response)
            detected_template = SUPPORTED_TEMPLATES[template_id]
        except:
            print("Did not get a valid response from the LLM.")
            return None

    return (
        detected_template.from_raw_types(target_column, dataframe)
        if detected_template
        else None
    )


def detect_template(
    dataset_path: str, target: str, task: str = None, openai_key: str = None, **kwargs
):
    if openai_key is not None and task is not None:
        print("Task detection using natural language enabled\n")
    openai_client = OpenAI(api_key=openai_key) if openai_key else None

    df = pd.read_csv(dataset_path).dropna().astype(str)
    verify_dataframe(df, target, task)

    detected_template = (
        get_template_from_query(
            query=task, openai_client=openai_client, target_column=target, dataframe=df
        )
        if task is not None
        else None
    )
    if detected_template is None:
        print("Enabling auto-inference on the dataframe.")
        detected_template = auto_infer_task_template(
            target_column_name=target, dataframe=df
        )

    template_names = "\n".join(f"• {name}" for name in SUPPORTED_TASK_TYPES)
    print(
        f"Task detected: {detected_template.task}\n"
        f"If this isn't the task you intended, you can:\n"
        + TEMPLATE_DETECTION_MESSAGE.format(
            dataset_path=dataset_path, target=target, template_names=template_names
        )
    )

    return detected_template
