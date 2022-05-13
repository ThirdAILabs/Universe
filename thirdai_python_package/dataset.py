import thirdai._thirdai.dataset
import thirdai._dataset_python.implementations
from thirdai._thirdai.dataset import *
from thirdai._dataset_python.implementations import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.dataset))
__all__.extend(dir(thirdai._dataset_python.implementations))

def load_text_classification_dataset(
    file: str,
    delim: str = "\t",
    labeled: bool = True,
    label_dim: int = 2,
    batch_size: int = 1024,
    encoding_dim: int = 100_000,
):

    # Define source
    source = sources.LocalFileSystem(file)
    parser = parsers.CsvIterable(delimiter="\t")

    # Define schema
    text_block = blocks.Text(col=1, dim=100000)
    label_block = blocks.Categorical(col=0, dim=2)
    schema = Schema(input_blocks=[text_block], target_blocks=[label_block])

    # Assemble
    loader = Loader(source, parser, schema, batch_size)

    return loader.processInMemory(), loader.input_dim()

__all__.append(load_text_classification_dataset)