class Callback:
    """Parent class for callbacks. Should never be instantiated.

    Attributes:
        metric (string): Quantity to be monitored. Currently, only categorical_accuracy is supported.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement, i.e.
            an absolute change of less than min_delta, will count as no improvement.
        baseline (float): Baseline value for the monitored quantity. Training will stop if the model
            doesn't show improvement over the baseline. Defaults to the monitored quantity
            observed on the first epoch.
        init_patience (int): Number of epochs with no improvement after which training will be stopped.
        patience (integer): Current number of epochs after which training will be stopped
            (patience <= init_patience).
        verbose (bool): False is silent, True displays a message if/when the callback takes effect.
    """

    def __init__(self, metric, verbose):
        self._metric = metric
        self._verbose = verbose

    def getMetric(self):
        return self._metric
    
    def callback(self, epoch, lr, epoch_metrics):
        return False, lr


class CallbackWithPatience(Callback):
    """Parent class for callbacks with patience. Should never be instantiated.

    Attributes:
        metric (string): Quantity to be monitored. Currently, only categorical_accuracy is supported.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement, i.e.
            an absolute change of less than min_delta, will count as no improvement.
        baseline (float): Baseline value for the monitored quantity. Training will stop if the model
            doesn't show improvement over the baseline. Defaults to the monitored quantity
            observed on the first epoch.
        init_patience (int): Number of epochs with no improvement after which training will be stopped.
        patience (integer): Current number of epochs after which training will be stopped
            (patience <= init_patience).
        verbose (bool): False is silent, True displays a message if/when the callback takes effect.
    """

    def __init__(self, metric, min_delta, baseline, patience, verbose):
        super().__init__(metric, verbose)
        self._min_delta = min_delta
        self._baseline = baseline
        self._patience = patience
        self._init_patience = patience

        if self._baseline == None:
            self._patience = (
                self._patience + 1
            )  # No baseline specified, increase initial patience to account for determining baseline
    
    def getBaseline(self):
        return self._baseline

    def getMinDelta(self):
        return self._min_delta

    def setBaseline(self, baseline):
        self._baseline = baseline

    def setDefaultMinDelta(self):
        self._min_delta = 0.001

    def callback(self, epoch, lr, epoch_metrics):
        """Returns flag indicating if training should be stopped due to lack of improvement.

        Implementation of standard callback function. Given the metrics for the most recent epoch,
        callback returns a boolean indicating whether training should be stopped and a learning
        rate.

        Args:
            epoch (integer): Unused. The epoch index (indexed from 0).
            lr (float): The current learning rate.
            epoch_metrics: The metrics for the most recent epoch as returned by Network.train().

        Returns:
            A tuple of (boolean, float) indicating whether training should be stopped and a learning
            rate (unchanged).
        """

        result = epoch_metrics[self._metric][
            0
        ]  # Get specified metric value for previous epoch
        if result - self._baseline < self._min_delta:
            self._patience -= 1
            if self._patience == 0:
                return self.onZeroPatience(lr)
            else:
                return False, lr
        else:
            self._baseline = (
                result  # Update baseline to reflect highest recorded metric
            )
            self._patience = self._init_patience  # Showing improvement, reset patience
            return False, lr

    def onZeroPatience(self, lr):
        return False, lr


class EarlyStop(CallbackWithPatience):
    """Monitors a metric and terminates training if no improvement is observed after a specified
        number of epochs.

    Attributes:
        metric (string): Quantity to be monitored. Currently, only categorical_accuracy is supported.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement, i.e.
            an absolute change of less than min_delta, will count as no improvement.
        baseline (float): Baseline value for the monitored quantity. Training will stop if the model
            doesn't show improvement over the baseline. Defaults to the monitored quantity
            observed on the first epoch.
        patience (integer): Number of epochs with no improvement after which training will be stopped.
        verbose (bool): False is silent, True displays a message if/when EarlyStop takes effect.
    """

    def __init__(
        self,
        metric="categorical_accuracy",
        min_delta=None,
        baseline=None,
        patience=0,
        verbose=True,
    ):
        super().__init__(metric, min_delta, baseline, patience, verbose)

    def onZeroPatience(self, lr):
        """Returns a boolean flag to halt training.

        Args:
            lr (unused): The current learning rate.

        Returns:
            A tuple of (boolean, float) indicating whether training should be stopped (always True)
            and a learning rate (unchanged).
        """

        if self._verbose:
            print(
                f"EarlyStop halted training after failing to meet improvement threshold for past {self._init_patience} epochs."
            )
        return True, lr


class AdaptiveLearningRate(CallbackWithPatience):
    """Monitors a metric and exponentially reduces learning rate if no improvement is observed after
        a specified number of epochs.

    Attributes:
        metric (string): Quantity to be monitored. Currently, only categorical_accuracy is supported.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement, i.e.
            an absolute change of less than min_delta, will count as no improvement.
        baseline (float): Baseline value for the monitored quantity. Training will stop if the model
            doesn't show improvement over the baseline. Defaults to the monitored quantity
            observed on the first epoch.
        patience (integer): Number of epochs with no improvement after which training will be stopped.
        verbose (bool): False is silent, True displays a message if/when AdaptiveLearningRate updates
            the learning rate.
    """

    def __init__(
        self,
        metric="categorical_accuracy",
        min_delta=None,
        baseline=None,
        patience=0,
        verbose=True,
    ):
        super().__init__(metric, min_delta, baseline, patience, verbose)

    def onZeroPatience(self, lr):
        """Returns an updated learning rate.

        Args:
            lr (float): The current learning rate.

        Returns:
            A tuple of (boolean, float) indicating whether training should be stopped (always False)
            and a learning rate.
        """

        _lr = lr * 10**-1
        self._patience = self._init_patience
        if self._verbose:
            print(
                f"AdaptiveLearningRate reduced learning rate from {lr} to {_lr} after failing to meet improvement threshold for past {self._init_patience} epochs."
            )
        return False, _lr  # Reduce learning rate exponentially


class LearningRateScheduler(Callback):
    """Modifies the learning rate after every epoch via a user-supplied function.

    Attributes:
        schedule (function): A function that takes an epoch index (integer, indexed from 0) and
            current learning rate (float) as inputs and returns a new learning rate as output (float).
        verbose (bool): False is silent, True displays a message when LearningRateScheduler updates the
            learning rate.
        Unused:
            metric (string): Quantity to be monitored. Currently, only categorical_accuracy is supported.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement, i.e.
                an absolute change of less than min_delta, will count as no improvement.
            baseline (float): Baseline value for the monitored quantity. Training will stop if the model
                doesn't show improvement over the baseline. Defaults to the monitored quantity
                observed on the first epoch.
            patience (integer): Number of epochs with no improvement after which training will be stopped.
    """

    def __init__(
        self,
        schedule,
        metric="categorical_accuracy",
        verbose=True,
    ):
        super().__init__(metric, verbose)
        self._schedule = schedule

    def callback(self, epoch, lr, epoch_metrics):
        """Returns a modified learning rate via a user-supplied function.

        Implementation of standard callback function. Given the current learning rate, callback
        modifies and returns it according to a user-supplied function.

        Args:
            epoch (integer): The epoch index (indexed from 0) which will be passed to the schedule.
            lr (float): The current learning rate which will be passed to the schedule.
            epoch_metrics: Unused. The metrics for the most recent epoch as returned by Network.train().

        Returns:
            A tuple of (boolean, float) indicating whether training should be stopped (always False)
            and a new learning rate.
        """

        _lr = self._schedule(epoch, lr)
        print(f"LearningRateScheduler updated learning rate from {lr} to {_lr}.")
        return False, _lr
