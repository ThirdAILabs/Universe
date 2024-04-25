from typing import List

import openai
from tqdm import tqdm


class QAGenMethod:
    def generate(self, texts: List[str]) -> List[List[str]]:
        pass


class OpenAI(QAGenMethod):
    def __init__(
        self, api_key: str, model: str, questions_per_paragraph: int, verbose=True
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.prompt = (
            f"Generate {questions_per_paragraph} questions from the "
            "following text. Try to ask questions that don't just reference "
            "keywords in the text. Return your answers as newline separated responses "
            "without any number or bullet prefixes. Here is the content: \n\n"
        )
        self.verbose = verbose

    def generate(self, texts: List[str]) -> List[List[str]]:
        questions = []
        for text in tqdm(texts, disable=not self.verbose):
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": self.prompt + text}],
                model=self.model,
            )
            response = chat_completion.choices[0].message.content
            questions.append(response.split("\n"))
        return questions
