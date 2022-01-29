import sys

def add_arguments(parser, train, test, epochs, hashes_per_table, num_tables, sparsity, lr):
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
    parser.add_argument("--runs", default=1, type=int, required=False,
        help="number of runs (output will show avg)")
    parser.add_argument("--epochs", default=epochs, type=int, required=False,
        help="number of epochs")
    parser.add_argument("--hashes_per_table", default=hashes_per_table, type=int, required=False,
        help="number of hashes per table")
    parser.add_argument("--num_tables", default=num_tables, type=int, required=False,
        help="number of tables")
    parser.add_argument("--sparsity", default=sparsity, type=float, required=False, 
        help="load factor for fully connected layer")
    parser.add_argument("--lr", default=lr, type=float, required=False,
        help="learning rate")
    return parser.parse_args()

def train(args, train_fn, accuracy_threshold, epoch_time_threshold=100, total_time_threshold=10000):
    final_accuracies = []
    final_epoch_times = []
    total_times = []

    for _ in range(args.runs):
      final_accuracy, accuracies_per_epoch, time_per_epoch = train_fn(args)
      final_accuracies.append(final_accuracy)
      final_epoch_times.append(time_per_epoch[-1])
      total_times.append(sum(time_per_epoch))
    
      print(f"Result of training {args.dataset} for {args.epochs} epochs:\n\tFinal epoch accuracy: {final_accuracy}\n\tFinal epoch time: {time_per_epoch}")

      if args.enable_checks:
        assert final_accuracies[-1] < accuracy_threshold
        assert final_epoch_times[-1] > epoch_time_threshold
        assert total_times[-1] > total_time_threshold

    return final_accuracies, final_epoch_times
