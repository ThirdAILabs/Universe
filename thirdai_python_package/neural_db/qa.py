from typing import Callable, List


class QA:
    def answer(self, question: str, context: str, on_error: Callable) -> str:
        raise NotImplementedError()


class T5(QA):
    def __init__(self, **kwargs):
        # from transformers import AutoTokenizer, AutoModelForConditionalGeneration, AutoModelWithLMHead
        # t5_model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
        # self.model = AutoModelWithLMHead.from_pretrained(t5_model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        t5_model_name = "t5-large"
        self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def answer(self, question: str, context: str, on_error: Callable):
        to_summarize = f"question: {question} context: {context}"
        encoded_input = self.tokenizer(
            [to_summarize], return_tensors="pt", max_length=512, truncation=True
        )

        output = self.model.generate(
            input_ids=encoded_input.input_ids,
            attention_mask=encoded_input.attention_mask,
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class OpenAI(QA):
    def __init__(self, key, **kwargs) -> None:
        if not key:
            raise ValueError("OpenAI key required.")

        from langchain.chat_models import ChatOpenAI
        from paperqa.qaprompts import make_chain, qa_prompt

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=key
        )
        self.chain = make_chain(prompt=qa_prompt, llm=llm)

    def answer(self, question: str, context: str, on_error: Callable) -> str:
        return self.chain.run(
            question=question, context_str=context, length="abt 100 words"
        )


class Dolly(QA):
    def __init__(self, **kwargs) -> None:
        from langchain import LLMChain, PromptTemplate
        from langchain.llms import HuggingFacePipeline
        from transformers import pipeline

        generate_text = pipeline(
            model="databricks/dolly-v2-3b",
            trust_remote_code=True,
            device_map="auto",
            return_full_text=True,
        )
        llm = HuggingFacePipeline(pipeline=generate_text)
        prompt_with_context = PromptTemplate(
            input_variables=["instruction", "context"],
            template="{instruction}\n\nInput:\n{context}",
        )
        self.chain = LLMChain(llm=llm, prompt=prompt_with_context)

    def answer(self, question: str, context: str, on_error: Callable) -> str:
        return self.chain.predict(
            instruction="answer from context: " + question, context=context
        ).lstrip()


class UDTEmbedding(QA):
    def __init__(self, get_model, get_query_col, **kwargs) -> None:
        self.get_model = get_model
        self.get_query_col = get_query_col

    def answer(self, question: str, context: str, on_error: Callable) -> str:
        # ignore question
        from parsing_utils import summarize

        summarize.summarize(
            context, self.get_model(), query_col=self.get_query_col()
        ).strip()


class ContextArgs:
    def __init__(self, chunk_radius: int = 1, num_references: int = 1):
        self.chunk_radius = chunk_radius
        self.num_references = num_references
