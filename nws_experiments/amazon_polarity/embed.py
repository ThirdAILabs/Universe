from utils import run_and_time

def main(num_cpus):
    print("Downloading dataset...")
    from datasets import load_dataset
    ds = load_dataset("mteb/amazon_polarity", "en")

    print("Downloading model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
    
    print("Preparing multi process pool...")
    pool, _ = run_and_time(lambda: model.start_multi_process_pool(['cpu'] * num_cpus))

    print("Embedding train text...")
    train_embeddings, _ = run_and_time(lambda: model.encode_multi_process(ds["train"]["text"], pool))
    print("Embedding val text...")
    val_embeddings, _ = run_and_time(lambda: model.encode_multi_process(ds["validation"]["text"], pool))
    print("Embedding test text...")
    test_embeddings, _ = run_and_time(lambda: model.encode_multi_process(ds["test"]["text"], pool))

    print("Closing pool...")
    model.stop_multi_process_pool(pool)

    print("Saving embeddings...")
    train_embeddings.dump("train_embeddings.np")
    val_embeddings.dump("val_embeddings.np")
    test_embeddings.dump("test_embeddings.np")


if __name__ == "__main__":
    import sys
    main(num_cpus=int(sys.argv[1]) if len(sys.argv) > 1 else 2)