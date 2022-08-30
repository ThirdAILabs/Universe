import thirdai._thirdai.dataset
from thirdai._thirdai.dataset import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.dataset))


def tokenize_to_svm(
    input_file, output_dim=100_000, output_file="preprocessed_data.svm"
):
    """Utility function that converts text datasets into vector representations, saves them in SVM format.\n\n
    Arguments:\n
     * input_file: String - Path to a text dataset. More on this below.\n
     * output_dim: Int (positive, optional) - The dimension of the vector representations
     produced by this function. Defaults to 100,000.\n
     * output_file: String - Path to a file where we save the vector representations.
     Defaults to ./preprocessed_data.svm\n\n

    **Text dataset format**\n
    The text dataset must be a CSV file where each row follows this format:\n\n

    \<pos or neg\>,\<text\> \n\n

    For example, we can have a training corpus called example_train.csv that contains the following:\n
    ```\n
    pos,Had a great time at the webinar.\n
    neg,I hate slow deep learning models.\n
    ```\n
    """
    import csv
    import re

    if input_file.find(".csv") == -1:
        raise ValueError("Only .csv files are supported")

    with open(output_file, "w") as fw:
        csvreader = csv.reader(open(input_file, "r"))

        for line in csvreader:
            label = 1 if line[0] == "pos" else 0
            fw.write(str(label) + " ")

            sentence = re.sub(r"[^\w\s]", "", line[1])
            sentence = sentence.lower()
            ### BOLT TOKENIZER START
            tup = thirdai._thirdai.dataset.bolt_tokenizer(
                sentence, dimension=output_dim
            )
            for idx, val in zip(tup[0], tup[1]):
                fw.write(str(idx) + ":" + str(val) + " ")
            ### BOLT TOKENIZER END

            fw.write("\n")


__all__.append(tokenize_to_svm)
