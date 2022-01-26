from thirdai import bolt, dataset
import numpy as np

from helpers import add_arguments, train

data = np.random.randn(640000,100)
batches = np.reshape(dataset, (-1, 64, 100))
bolt_dataset = dataset.makeInMemoryDatasetFromBatches([dataset.wrapNumpyIntoDenseBatch(batch, i * 64) for i, batch in enumerate(batches)])
bolt

