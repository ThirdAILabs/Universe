from modeling.checkpoint import Checkpoint


class Colbert:
    def __init__(self, checkpoint_path):
        self.checkpoint = Checkpoint(checkpoint_path).cpu()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query]).numpy()[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs).numpy()
