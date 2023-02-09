import os
from argparse import ArgumentParser

from thirdai.dataset import Wordpiece
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vocab-dir", type=str, required=True)
    parser.add_argument("--lower", action="store_true")

    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--expected-lines", type=int, default=None)

    parser.add_argument("--log", type=str, required=True)

    args = parser.parse_args()

    tag = "bert-base-uncased" if args.lower else "bert-base-cased"
    vocab_path = os.path.join(args.vocab_dir, f"{tag}.txt")

    thirdai_tokenizer = Wordpiece.make(vocab_path, args.lower)
    hf_tokenizer = AutoTokenizer.from_pretrained(tag)

    with open(args.corpus_path) as fp:
        with open(args.log, "w+") as log_fp:
            wrapper = (
                lambda x: tqdm(x, total=args.expected_lines)
                if args.expected_lines is not None
                else lambda x: x
            )
            for idx, line in enumerate(wrapper(fp)):
                line = line.strip()

                hf_ids = hf_tokenizer.encode(line)
                assert len(hf_ids) >= 2  # bos, eos is expected.
                # Trim both eos, bos. Keep unk, which is also a special token.
                hf_ids = hf_ids[1 : len(hf_ids) - 1]
                hf_tokens = hf_tokenizer.convert_ids_to_tokens(hf_ids)
                hf_decoded = " ".join(hf_tokens)

                thirdai_ids = thirdai_tokenizer.encode(line)
                thirdai_decoded = thirdai_tokenizer.decode(thirdai_ids)

                if hf_ids != thirdai_ids:
                    print(idx, line, file=log_fp)
                    print("hft-ids", hf_ids, file=log_fp)
                    print("tai-ids", thirdai_ids, file=log_fp)
                    print("hft-dec", hf_decoded, file=log_fp)
                    print("tai-dec", thirdai_decoded, file=log_fp)
