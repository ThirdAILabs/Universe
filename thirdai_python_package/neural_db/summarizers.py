class Summarizer:
    def summarize(self, context: str) -> str:
        raise NotImplementedError()


class UDTSummarizer(Summarizer):
    def __init__(self, get_model, get_query_col, **kwargs) -> None:
        self.get_model = get_model
        self.get_query_col = get_query_col

    def summarize(self, context: str, **kwargs) -> str:
        from .summarizer_utils import udt_summarize

        summary = udt_summarize.summarize(
            context, self.get_model(), query_col=self.get_query_col()
        ).strip()

        return " ".join(
            [sent.strip() for sent in udt_summarize.nlkt_sent_tokenize(summary)]
        )