from modeling.checkpoint import Checkpoint


class Colbert:
    def __init__(self, checkpoint_path):
        self.checkpoint = Checkpoint(checkpoint_path).cpu()
        print(self.checkpoint.colbert_config)

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encodeQuery(self, queries):
        return self.checkpoint.queryFromText(queries)

    def encodeDoc(self, docs):
        return self.checkpoint.docFromText(docs)
