from ..cookie_monster import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir", type=str, required=True, help="Path to training data"
)
parser.add_argument("--test_dir", type=str, required=True, help="Path to test data")
parser.add_argument(
    "--command",
    type=str,
    default="classification",
    help="Accepted commands are mlm or classification",
)
parser.add_argument("--input_dim", type=int, default=100000, help="Input dimension")
parser.add_argument("--output_dim", type=int, default=2, help="Output dimension")

args = parser.parse_args()


def construct_monster(output_dim):
    model = CookieMonster(
        args.input_dim, hidden_dimension=2000, hidden_sparsity=0.1, mlflow_enabled=False
    )
    model.set_output_dimension(output_dim, args.command)
    return model


if "__main__" == __name__:
    if args.command == "classification":
        model = construct_monster(args.output_dim)
    elif args.command == "mlm":
        model = construct_monster(30224)
    else:
        raise ValueError("Invalid command. Supported commands are: classification, mlm")

    model.eat_corpus(args.train_dir, False, True)
    model.evaluate(args.test_dir)
