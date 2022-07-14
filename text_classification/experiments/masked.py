from ..cookie_monster import *
import sys


def preprocess_to_mlm(original_file, mlm_file):
    pass


if len(sys.argv) != 3:
    print("Invalid args: Usage python3 <script>.py <train_dir> <test_dir>")
    sys.exit(1)

TRAIN_DIR = sys.argv[1]
TEST_DIR = sys.argv[2]
INPUT_DIM = 100000
VOCAB_SIZE = 30224

model = CookieMonster(
    INPUT_DIM,
    hidden_dimension=2000,
    output_dimension=VOCAB_SIZE,
    hidden_sparsity=0.1,
    mlflow_enabled=True,
)
model.eat_corpus(TRAIN_DIR, True, True)
model.evaluate(TEST_DIR)
