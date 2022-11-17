import time

import thirdai._thirdai.dataset
from thirdai._thirdai.dataset import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.dataset))


class S3DataLoader(DataLoader):
    def __init__(
        self,
        bucket_name,
        prefix_filter,
        batch_size,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        DataLoader.__init__(self, batch_size)

        # We are doing this import here instead of at the top of the file
        # so boto3 is not a dependency of our package
        import boto3

        if aws_access_key_id:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            self._s3_client = session.resource("s3")
        else:
            self._s3_client = boto3.resource("s3")

        self._batch_size = batch_size
        self._bucket_name = bucket_name
        self._bucket = self._s3_client.Bucket(self._bucket_name)
        self._prefix_filter = prefix_filter
        self._objects_in_bucket = list(
            self._bucket.objects.filter(Prefix=prefix_filter)
        )
        self.restart()

    def restart(self):
        self._line_iterator = self._get_line_iterator()

    def _get_line_iterator(self):
        for obj in self._objects_in_bucket:
            key = obj.key
            print("Now parsing object " + key)
            body = obj.get()["Body"]
            for line in body.iter_lines():
                yield line

    def next_batch(self):
        lines = []
        while len(lines) < self._batch_size:
            next_line = self.next_line()
            if next_line == None:
                break
            lines.append(next_line)
        if lines == []:
            return None
        return lines

    def next_line(self):
        next_line = next(self._line_iterator, None)
        if next_line:
            next_line = next_line.decode("utf-8")
        return next_line

    def resource_name(self):
        return f"s3://{self._bucket_name}/{self._prefix_filter}"


__all__.append("S3DataLoader")
