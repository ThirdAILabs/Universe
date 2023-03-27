from thirdai import bolt

class ColdStartCallback(bolt.callbacks.Callback):
    def __init__(self, UDT, save_loc, metric, n_bad_batches_before_update, n_lr_updates):
        super().__init__()
        self.model = UDT
        self.save_loc = save_loc
        self.metric = metric
        self.total_bad_batches_before_update = n_bad_batches_before_update
        self.n_bad_batches = 0
        self.total_n_lr_updates = n_lr_updates
        self.n_lr_updates = 0
        self.scaledown = 2
        self.best_metric = 0
        self.last_metric = 0

    def on_batch_end(self, model, train_state):
        cur_metric = train_state.get_all_train_batch_metrics()[self.metric][-1]

        if cur_metric > self.best_metric:
            self.model.save(self.save_loc)
            self.best_metric = cur_metric

        if cur_metric < self.last_metric:
            self.n_bad_batches += 1
        else:
            self.n_bad_batches = 0
        
        if self.n_bad_batches > self.total_bad_batches_before_update:
            if self.n_lr_updates > self.total_n_lr_updates:
                train_state.stop_training = True
            else:
                train_state.learning_rate /= self.scaledown
                self.n_lr_updates += 1

        self.last_metric = cur_metric
