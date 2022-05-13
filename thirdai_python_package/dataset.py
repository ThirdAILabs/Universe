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
    parser = parsers.CsvIterable(delimiter=delim)

    # Define schema
    if labeled:
        label_block = blocks.Categorical(col=0, dim=label_dim)
        text_block = blocks.Text(col=1, dim=encoding_dim)
        schema = Schema(input_blocks=[text_block], target_blocks=[label_block])
    else:
        # Text in first column if no label.
        text_block = blocks.Text(col=0, dim=encoding_dim)
        schema = Schema(input_blocks=[text_block])

    # Assemble
    loader = Loader(source, parser, schema, batch_size)

    return loader.processInMemory(), loader.input_dim()

__all__.append(load_text_classification_dataset)