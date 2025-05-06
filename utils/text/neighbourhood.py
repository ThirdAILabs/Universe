from thirdai import data
import pandas as pd

import time
import tqdm

import argparse

import thirdai

thirdai.set_global_num_threads(10)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_parallelized", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--no_stemming", action="store_true")
    return parser.parse_args()


def convert_column_to_sentences(df, column):
    return list(df[column])


def timed(func):
    def implementor(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        return {"value": ret, "time": end - start}

    return implementor


@timed
def load_file(filename, columns, args):
    df = pd.read_csv(filename)

    df = df.sample(frac=0.1)
    print(len(df))

    sentences = []
    for c in columns:
        sentences += convert_column_to_sentences(df, c)

    if not args.no_stemming:
        print("Using stemmer")

    neighbourhood_tracker = data.CollocationTracker(args.topk, not args.no_stemming)

    if not args.use_parallelized:
        print("Using sequential code")
        for sentence in tqdm.tqdm(sentences):
            neighbourhood_tracker.index_sentence(sentence)
    else:
        print("Using paralle code")
        for chunk_id in tqdm.tqdm(range(len(sentences) // args.chunk_size + 1)):
            chunk_sentence = sentences[
                chunk_id * args.chunk_size : (chunk_id + 1) * args.chunk_size
            ]
            neighbourhood_tracker.index_sentences(chunk_sentence)

    return neighbourhood_tracker


if __name__ == "__main__":
    args = parse_args()

    import json

    print(json.dumps(vars(args), indent=4))

    processed_data = load_file(
        "/home/shubh/Universe/scifact/pubmed_1M.csv", ["abstract", "title"], args
    )

    print("time_taken: ", processed_data["time"])

    tracker = processed_data["value"]
    dict = {}
    dict["processing_time"] = processed_data["time"]

    # for word in tracker.get_words():
    #     dict[word] = tracker.get_neighbours_counter(word).get_closest_neighbours()

    # with open("collocation_serialized.json", "w") as f:
    #     json.dump(dict, f, indent=2)

    print(tracker.get_neighbours_counter("diabetes").get_closest_neighbours()[:10])

    print(tracker.get_neighbours_counter("insulin").get_closest_neighbours()[:10])

    print(tracker.get_neighbours_counter("blood").get_closest_neighbours()[:10])
