def add_arguments(parser, train_data, test_data, K, L, sparsity, lr):
    """
    Add arguments to each benchmark script.
    """
    parser.add_argument(
        "--train",
        default=train_data,
        help="file path of train data",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--test",
        default=test_data,
        help="file path of test data",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--K", default=K, help="number of hashes per table", type=int, required=False
    )
    parser.add_argument(
        "--L", default=L, help="number of tables", type=int, required=False
    )
    parser.add_argument(
        "--sparsity",
        default=sparsity,
        help="load factor for fully connected layer",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lr", default=lr, help="learning rate", type=float, required=False
    )
    return parser.parse_args()
