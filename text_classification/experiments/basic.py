from ..cookie_monster import *
import argparse

INPUT_DIM = 100000
VOCAB_SIZE = 30224

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--command", type=str, default="classification")
parser.add_argument("--output_dim", type=int, default=2)

args = parser.parse_args()


def construct_monster(output_dim):
    model = CookieMonster(
        INPUT_DIM, hidden_dimension=2000, hidden_sparsity=0.1, mlflow_enabled=False
    )
    model.set_output_dimension(output_dim)
    return model


if "__main__" == __name__:
    if args.command == "classification":
        model = construct_monster(args.output_dim)
    elif args.command == "mlm":
        model = construct_monster(VOCAB_SIZE)
    else:
        raise ValueError("Invalid command. Supported commands are: classification, mlm")

    model.eat_corpus(args.train_dir, False, True)
    model.evaluate(args.test_dir)
