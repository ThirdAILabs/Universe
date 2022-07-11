from numpy import isin
import thirdai._thirdai.bolt
from thirdai._thirdai.bolt import *

import thirdai._callbacks.callbacks
from thirdai._callbacks import callbacks
from thirdai._callbacks.callbacks import *

import thirdai._bolt_datastructures
from thirdai._bolt_datastructures import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.bolt))
__all__.extend(dir(thirdai._callbacks.callbacks))
__all__.extend(dir(thirdai._bolt_datastructures))


class Network(thirdai._thirdai.bolt.Network):
    def train(
        self,
        train_data,
        train_labels,
        loss_fn,
        learning_rate,
        epochs,
        batch_size=0,
        rehash=0,
        rebuild=0,
        verbose=True,
        metrics=[],
        callbacks=[],
    ):
        lr = learning_rate
        _metrics = metrics
        stop_flag = False
        training_history = {}

        for (
            callback
        ) in callbacks:  # Add metrics required by callbacks, if not already present
            if callback.getMetric() not in metrics:
                _metrics.append(callback.getMetric())

        for epoch in range(epochs):
            if stop_flag:  # Early stoppage
                return training_history

            epoch_metrics = thirdai._thirdai.bolt.Network.train(
                self,
                train_data=train_data,
                train_labels=train_labels,
                loss_fn=loss_fn,
                learning_rate=lr,
                epochs=1,
                batch_size=batch_size,
                rehash=rehash,
                rebuild=rebuild,
                verbose=verbose,
                metrics=metrics,
            )

            if epoch == 0:
                training_history = epoch_metrics  # Begin compiling history
                for (
                    callback
                ) in callbacks:  # Initialize baselines and min_deltas, if necessary
                    if callback.getBaseline() == None:
                        callback.setBaseline(epoch_metrics[callback.getMetric()][0])
                    if callback.getMinDelta() == None:
                        callback.setDefaultMinDelta()
            else:
                for (
                    key,
                    value,
                ) in (
                    epoch_metrics.items()
                ):  # Add metrics from most recent epoch to complete training history
                    training_history[key].extend(value)

            for callback in callbacks:  # Call callbacks
                stop_flag, lr = callback.callback(epoch, lr, epoch_metrics)

        return training_history

