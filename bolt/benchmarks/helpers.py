import sys

def add_arguments(parser, train, test, epochs, K, L, sparsity, lr):
    """
    Add arguments to each benchmark script.
    """
    parser.add_argument("dataset", type=str, help="dataset name")
    parser.add_argument("--train", default=train, type=str, required=False,
        help="file path of train data")
    parser.add_argument("--test", default=test, type=str, required=False,
        help="file path of test data")
    parser.add_argument("--enable_checks", default=False, required=False,
        help="train with error checking", action='store_true')
    parser.add_argument("--epochs", default=epochs, type=int, required=False,
        help="number of epochs")
    parser.add_argument("--K", default=K, type=int, required=False,
        help="number of hashes per table")
    parser.add_argument("--L", default=L, type=int, required=False,
        help="number of tables")
    parser.add_argument("--sparsity", default=sparsity, type=float, required=False, 
        help="load factor for fully connected layer")
    parser.add_argument("--lr", default=lr, type=float, required=False,
        help="learning rate")
    return parser.parse_args()

def train(args, train_fn, accuracy_threshold, epoch_time_threshold=100, total_time_threshold=10000, max_runs=1):
    final_accuracies = []
    final_epoch_times = []
    total_times = []
    for _ in range(max_runs):
        final_accuracy, accuracies_per_epoch, time_per_epoch = train_fn(args)
        final_accuracies.append(final_accuracy)
        final_epoch_times.append(time_per_epoch[-1])
        total_times.append(sum(time_per_epoch))
        if args.enable_checks and final_accuracy > accuracy_threshold and time_per_epoch[-1] < epoch_time_threshold and sum(time_per_epoch) < total_time_threshold:
            print(f"Passed training checks for {args.dataset} on {args.epochs} epochs with:\n\tFinal epoch accuracies: {final_accuracies}\n\tFinal epoch times(s): {final_epoch_times}")
            return final_accuracies, final_epoch_times
    if args.enable_checks:
        if max(final_accuracies) < accuracy_threshold:
            print(f'Epoch {args.epochs} accuracy *({max(final_accuracies)})* is lower than expected *({accuracy_threshold})*')
        if min(final_epoch_times) > epoch_time_threshold:
            print(f'Epoch {args.epochs} training time *({min(final_epoch_times)}))* took longer than expected *({epoch_time_threshold})*')
        if min(total_times) > total_time_threshold:
            print(f'Training time took longer than expected *({min(total_times)})*')
        sys.exit(1)
    return final_accuracies, final_epoch_times
