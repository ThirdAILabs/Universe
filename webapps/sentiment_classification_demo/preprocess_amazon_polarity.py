import csv
import re
import thirdai
import argparse


def preprocess_amazon_polarity(input_file, output_dim, output_file, has_header=True):
    """
    Helper function to preprocess the amazon polarity dataset for training.
    This assumes that the header of the dataset has been removed.
    """

    if not input_file.endswith(".csv"):
        raise ValueError("Only .csv files are supported")

    with open(output_file, "w") as fw:
        csvreader = csv.reader(open(input_file, "r"))

        first_line = True
        for line in csvreader:
            if has_header and first_line:
                first_line = False
                continue
            if len(line) != 2:
                raise ValueError("Expcted csv to have 2 columns per line")

            label = int(line[1])

            fw.write(str(label) + " ")

            non_word_or_whitespace = r"[^\w\s]"
            sentence = re.sub(non_word_or_whitespace, "", line[0])
            sentence = sentence.lower()
            # BOLT TOKENIZER START
            tup = thirdai.dataset.bolt_tokenizer(
                sentence, seed=341, dimension=output_dim
            )
            for idx, val in zip(tup[0], tup[1]):
                fw.write(str(idx) + ":" + str(val) + " ")
            # BOLT TOKENIZER END

            fw.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess amazon polarity")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help='Path to the dataset.',
    )
    parser.add_argument(
        "-d",
        "--output_dim",
        type=int,
        required=True,
        help='Dimension of output.',
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help='Output filename.',
    )
    parser.add_argument(
        "-a",
        "--has_header",
        action="store_true",
        help='Dimension of output.',
    )

    args = parser.parse_args()

    preprocess_amazon_polarity(
        args.input_file, args.output_dim, args.output_file, args.has_header)


if __name__ == "__main__":
    main()
