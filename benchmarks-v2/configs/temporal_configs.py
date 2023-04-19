from thirdai import bolt

from .udt_configs import UDTBenchmarkConfig


class MovieLensUDTBenchmark(UDTBenchmarkConfig):
    config_name = "movie_lens_temporal"
    dataset_name = "movie_lens"

    train_file = "movielens1m/train.csv"
    test_file = "movielens1m/test.csv"

    target = "movieId"
    n_target_classes = 3706
    temporal_relationships = {
        "userId": [
            bolt.temporal.categorical(column_name="movieId", track_last_n=length)
            for length in [1, 2, 5, 10, 25, 50]
        ]
    }

    learning_rate = 0.0001
    num_epochs = 5
    metrics = ["recall@10", "precision@10"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "userId": bolt.types.categorical(),
            "movieId": bolt.types.categorical(delimiter=" "),
            "timestamp": bolt.types.date(),
        }


class AmazonGamesUDTBenchmark(UDTBenchmarkConfig):
    config_name = "amazon_games_temporal"
    dataset_name = "amazon_games"

    train_file = "amazon_games/train.csv"
    test_file = "amazon_games/test.csv"

    target = "gameId"
    n_target_classes = 33388
    temporal_relationships = {
        "userId": [
            bolt.temporal.categorical(column_name="gameId", track_last_n=length)
            for length in [1, 2, 5, 10, 25]
        ]
    }

    learning_rate = 0.0001
    num_epochs = 50
    metrics = ["recall@10", "precision@10"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "userId": bolt.types.categorical(),
            "gameId": bolt.types.categorical(delimiter=" "),
            "timestamp": bolt.types.date(),
        }


class NetflixUDTBenchmark(UDTBenchmarkConfig):
    config_name = "netflix_temporal"
    dataset_name = "netflix_100M"

    train_file = "netflix/netflix_train.csv"
    test_file = "netflix/netflix_test.csv"

    target = "movie"
    n_target_classes = 17770
    temporal_relationships = {
        "user": [
            bolt.temporal.categorical(column_name="movie", track_last_n=length)
            for length in [1, 2, 5, 10, 25, 50]
        ]
    }

    learning_rate = 0.0001
    num_epochs = 5
    metrics = ["recall@10", "precision@10"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "user": bolt.types.categorical(),
            "movie": bolt.types.categorical(delimiter=" "),
            "date": bolt.types.date(),
        }
