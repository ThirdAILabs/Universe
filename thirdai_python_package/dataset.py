import thirdai._thirdai.dataset
from thirdai._thirdai.dataset import *
import time

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


class S3DataLoader(DataLoader):
    def __init__(
        self,
        s3_bucket,
        prefix_filter,
        batch_size,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        DataLoader.__init__(self, batch_size)

        import boto3

        if aws_access_key_id:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            self.s3_client = session.resource("s3")
        else:
            self.s3_client = boto3.resource("s3")

        self.batch_size = batch_size
        self.bucket = self.s3_client.Bucket(s3_bucket)
        self.prefix_filter = prefix_filter
        self.objects_in_bucket = list(self.bucket.objects.filter(Prefix=prefix_filter))
        self.line_iterator = self._get_line_iterator()
        self.current_batch_id = 0

    def _get_line_iterator(self):
        for obj in self.objects_in_bucket:
            key = obj.key
            print("Now parsing object " + key)
            body = obj.get()["Body"]
            for line in body.iter_lines():
                yield line

    def next_batch(self):
        print(
            "Loading new batch: " + str(self.current_batch_id), time.time(), flush=True
        )
        lines = []
        while len(lines) < self.batch_size:
            next_line = self.get_next_line()
            if next_line == None:
                break
            lines.append(next_line)
        print(
            f"Batch {self.current_batch_id} created with {len(lines)} elements",
            time.time(),
            flush=True,
        )
        self.current_batch_id += 1
        if lines == []:
            return None
        return lines

    def get_next_line(self):
        return next(self.line_iterator, None)

    def resource_name(self):
        return self.bucket + "/" + self.prefix_filter


__all__.append(tokenize_to_svm)
__all__.append(S3DataLoader)
