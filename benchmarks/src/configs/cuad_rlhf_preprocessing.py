import os
from collections import defaultdict

# This is to fix an error that occurred running on the aws machine.
import nltk
import numpy as np
import pandas as pd

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

MIN_WORDS_PER_CHUNK = 50
CHUNK_THRESHOLD = 150


def chunk_text(text: str):
    sentences = sent_tokenize(text)

    # We want to split hte text into phrases while preserving the original number
    # of characters in the text because answers in the cuad dataset reference a
    # location in the text using the character offset. We need the character count
    # to be consistent so that we can deterimine which phrase an answer occurs in.
    start = 0
    for i, _ in enumerate(sentences):
        end = start + text[start:].index(sentences[i]) + len(sentences[i])
        sentences[i] = text[start:end]
        start = end
    if start != len(text):
        if len(sentences) == 0:
            sentences.append("")
        sentences[-1] += text[start:]
    if len(sentences) == 1:
        return [text]

    words_per_sentence = [len(word_tokenize(sent)) for sent in sentences]
    if sum(words_per_sentence) < CHUNK_THRESHOLD:
        return [text]

    chunks = []
    cur_word_count = 0
    start_idx = 0

    for idx in range(len(sentences)):
        word_count = words_per_sentence[idx]
        if cur_word_count < MIN_WORDS_PER_CHUNK:
            cur_word_count += word_count
        else:
            chunks.append("".join(sentences[start_idx:idx]))
            start_idx = idx
            cur_word_count = word_count

    if start_idx != len(sentences):
        final_chunk = "".join(sentences[start_idx : len(sentences)])
        if len(chunks) > 0 and cur_word_count < MIN_WORDS_PER_CHUNK:
            chunks[-1] += final_chunk
        else:
            chunks.append(final_chunk)
    return chunks


def get_paragraphs(df_data):
    paras = [
        chunk
        for para in df_data["paragraphs"][0]["context"]
        # Some paragraphs are separated by '\n\n' which causes empty strings after split.
        # This ensures that the character count is preserved.
        .replace("\n\n", " \n").split("\n")
        for chunk in chunk_text(para + "\n")
    ]
    return paras


def process_cuad_data(
    cuad_dataset,
    paragraphs_to_answers_filename,
    association_samples_filename,
    questions_and_answers_filename,
    per_contract_data_dirname,
    contract_eval_filename,
    contract_paragraphs_filename,
):
    """
    This preprocessing creates several datasets:
    - paragraphs_to_answers: Text of paragraphs for all contracts and the ids of
      questions each paragraph contains answers for. Used for pretraining.

    - association_samples: Text of question along with text of answer/text of
      answer paragraph. Used for RLHF.

    - questions_and_answers: Text and id of each question along with the text of
      all of its answers across all of the documents. Used for cold start pretraining
      with the question as a strong column and its answers as a weak column.

    - contract_eval: An evaluation set, one per contract, which contains the text
      of questions and the ids of the paragraphs containing the answers.

    - contract_paragraphs: The text of each paragraph in a given contract, along with
      the id for that paragraph. This is used to introduce the contents of the contract
      before evaluating the accuracy on the contract.
    """
    df = pd.read_json(cuad_dataset)

    question_to_answers = defaultdict(list)
    paragraphs_to_questions = []
    association_samples = []
    per_contract_data = {}

    question_ids = {}

    def get_question_and_id(qa):
        question = qa["question"]
        question = question[question.index("Details: ") + 9 :]
        if question not in question_ids:
            question_ids[question] = len(question_ids)
        question_id = question_ids[question]
        return question, question_id

    for contract in df["data"]:
        assert len(contract["paragraphs"]) == 1
        title = contract["title"]

        paragraphs = get_paragraphs(contract)
        paragraph_offsets = np.cumsum([len(p) for p in paragraphs])

        evaluation_samples = []

        paragraph_ids_to_questions = defaultdict(set)

        for qa in contract["paragraphs"][0]["qas"]:
            if len(qa["answers"]) == 0:
                continue

            question, question_id = get_question_and_id(qa)

            answer_paragraph_ids = []
            for answer in qa["answers"]:
                answer_paragraph_id = np.where(
                    paragraph_offsets > answer["answer_start"]
                )[0][0]
                answer_paragraph = paragraphs[answer_paragraph_id]

                question_to_answers[question].append(answer["text"])

                association_sample = {
                    "source": question,
                    "target_answer": answer["text"],
                    "target_paragraph": answer_paragraph,
                }
                association_samples.append(association_sample)

                paragraph_ids_to_questions[answer_paragraph_id].add(question_id)

                answer_paragraph_ids.append(answer_paragraph_id)

            evaluation_samples.append(
                {"text": question, "id": ";".join(map(str, answer_paragraph_ids))}
            )

        for p_id, qs in paragraph_ids_to_questions.items():
            paragraphs_to_questions.append(
                {"text": paragraphs[p_id], "id": ";".join(map(str, qs))}
            )

        per_contract_data[title] = {
            "paragraphs": pd.DataFrame(
                {"text": paragraphs, "id": np.arange(len(paragraphs))}
            ),
            "eval": pd.DataFrame.from_records(evaluation_samples),
        }

    paragraphs_to_questions = pd.DataFrame.from_records(paragraphs_to_questions)
    paragraphs_to_questions.to_csv(paragraphs_to_answers_filename, index=False)

    association_samples = pd.DataFrame.from_records(association_samples)
    association_samples.to_csv(association_samples_filename, index=False)

    create_questions_and_answers_dataset(
        question_to_answers=question_to_answers,
        question_ids=question_ids,
        filename=questions_and_answers_filename,
    )

    create_per_contract_datasets(
        per_contract_data=per_contract_data,
        per_contract_data_dirname=per_contract_data_dirname,
        contract_eval_filename=contract_eval_filename,
        contract_paragraphs_filename=contract_paragraphs_filename,
    )


def create_questions_and_answers_dataset(question_to_answers, question_ids, filename):
    questions_answers = pd.DataFrame.from_records(
        [
            {
                "id": question_ids[question],
                "question": question,
                "answers": " ".join(answers),
            }
            for question, answers in question_to_answers.items()
        ]
    )
    questions_answers.to_csv(filename, index=False)


def create_per_contract_datasets(
    per_contract_data,
    per_contract_data_dirname,
    contract_eval_filename,
    contract_paragraphs_filename,
):
    for contract, data in per_contract_data.items():
        contract_dir = os.path.join(
            per_contract_data_dirname, contract.replace(" ", "_")
        )
        os.makedirs(contract_dir)
        data["eval"].to_csv(
            os.path.join(contract_dir, contract_eval_filename), index=False
        )
        data["paragraphs"].to_csv(
            os.path.join(contract_dir, contract_paragraphs_filename), index=False
        )
