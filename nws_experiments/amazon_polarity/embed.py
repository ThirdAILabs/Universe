from utils import run_and_time
import tqdm
import numpy as np
import os


def main(num_cpus):
    print("Downloading dataset...")
    from datasets import load_dataset
    ds = load_dataset("mteb/amazon_polarity", "default")

    print("Downloading model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
    
    print("Preparing multi process pool...")
    pool, _ = run_and_time(lambda: model.start_multi_process_pool(['cpu'] * num_cpus))

    def np_dump_size(filename):
        try:
            return len(np.load(filename, allow_pickle=True))
        except:
            return 0

    def embed_with_progress(data, save_to):
        batch_size = 3200
        for i in tqdm.tqdm(range(0, len(data), batch_size)):
            filename = f"{save_to}_{i}.np"
            batch = data[i:i+batch_size]
            if os.path.exists(filename) and (np_dump_size(filename) == len(batch)):
                continue
            embeddings = model.encode_multi_process(batch, pool, batch_size=32)
            embeddings.dump(f"{save_to}_{i}.np")

    print("Embedding test text...")
    run_and_time(lambda: embed_with_progress(
        data=ds["test"]["text"],
        save_to="/share/benito/nws/amazon_polarity/test_embeddings"
    ))
    print("Embedding train text...")
    run_and_time(lambda: embed_with_progress(
        data=ds["train"]["text"],
        save_to="/share/benito/nws/amazon_polarity/train_embeddings",
    ))
    
    print("Closing pool...")
    model.stop_multi_process_pool(pool)


if __name__ == "__main__":
    import sys
    main(num_cpus=int(sys.argv[1]) if len(sys.argv) > 1 else 2)