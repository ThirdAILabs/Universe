def add_arguments(
    parser, train, test, epochs, hashes_per_table, num_tables, sparsity, lr
):
    """
    Add arguments to each benchmark script.
    """
    parser.add_argument(
        "--train",
        default=train,
        type=str,
        required=False,
        help="file path of train data",
    )
    parser.add_argument(
        "--test", default=test, type=str, required=False, help="file path of test data"
    )
    parser.add_argument(
        "--runs",
        default=1,
        type=int,
        required=False,
        help="number of runs (output will show avg)",
    )
    parser.add_argument(
        "--epochs", default=epochs, type=int, required=False, help="number of epochs"
    )
    parser.add_argument(
        "--hashes_per_table",
        default=hashes_per_table,
        type=int,
        required=False,
        help="number of hashes per table",
    )
    parser.add_argument(
        "--num_tables",
        default=num_tables,
        type=int,
        required=False,
        help="number of tables",
    )
    parser.add_argument(
        "--sparsity",
        default=sparsity,
        type=float,
        required=False,
        help="load factor for fully connected layer",
    )
    parser.add_argument(
        "--lr", default=lr, type=float, required=False, help="learning rate"
    )
    return parser.parse_args()
