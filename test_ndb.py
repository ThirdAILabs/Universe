from thirdai import neural_db as ndb
from pathlib import Path
import pandas as pd
    

class CSVDocument(ndb.Document):
    def __init__(self, path, strong_columns, weak_columns, reference_columns):
        from thirdai.neural_db.utils import hash_file
        self.path = Path(path)
        self._hash = hash_file(path)
        self.df = pd.read_csv(path)
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = reference_columns
    def hash(self) -> str:
        return self._hash
    def size(self) -> int:
        return len(self.df)
    def name(self) -> str:
        return self.path.name
    def strong_text(self, element_id: int) -> str:
        row = self.df.iloc[element_id]
        return " ".join([row[col] for col in self.strong_columns])
    def weak_text(self, element_id: int) -> str:
        row = self.df.iloc[element_id]
        return " ".join([row[col] for col in self.weak_columns])
    def reference(self, element_id: int) -> ndb.Reference:
        row = self.df.iloc[element_id]
        text = " ".join([row[col] for col in self.reference_columns])
        def show_fn(*args, **kwargs):
            print("Showing", text)
        return ndb.Reference(element_id, text, str(self.path.absolute()), {}, show_fn)
    def context(self, element_id: int, radius) -> str:
        row = self.df.iloc[element_id]
        return " ".join([row[col] for col in self.reference_columns])
    def save_meta(self, directory: Path):
        raise NotImplementedError()
    def load_meta(self, directory: Path):
        raise NotImplementedError()

db = ndb.NeuralDB("")
db.from_scratch()
db.add_documents(
    [CSVDocument(
        "test_cs.csv", 
        weak_columns=["weak1", "wEaK2"], 
        strong_columns=["strong1", "stronk2"], 
        reference_columns=["ref1", "REF2"])],
    on_success= lambda: print("SUCCESS"),
    on_error=lambda error_msg: print(error_msg),
    on_irrecoverable_error=lambda error_msg: print(error_msg),
)
print(db.sources())
for result in db.search("STRONK", 2, on_error=lambda error_msg: print(error_msg)):
    print(result.text())

db.upvote(0)