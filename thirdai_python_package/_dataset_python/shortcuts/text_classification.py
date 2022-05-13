from thirdai.dataset import Loader, Schema, parsers, sources, blocks, text_encodings


def load_text_classification_dataset(
    file: str,
    delim: str = "\t",
    labeled: bool = True,
    label_dim: int = 2,
    batch_size: int = 1024,
    encoding_dim: int = 100_000,
):

    ######## DEFINE SOURCE ########
    source = sources.LocalFileSystem(file)
    parser = parsers.CsvIterable(delimiter="\t")

    ######## DEFINE SCHEMA ########
    text_block = blocks.Text(col=1, dim=100000)
    label_block = blocks.Categorical(col=0, dim=2)
    schema = Schema(input_blocks=[text_block], target_blocks=[label_block])

    ######## PUT TOGETHER ########
    dataset = Loader(source, parser, schema, batch_size)

    return dataset.processInMemory(), dataset.input_dim()
