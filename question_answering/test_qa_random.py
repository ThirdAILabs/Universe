import random
from ms_marco_eval import compute_metrics_from_files

result_file_name = "temp_ranking.txt"

with open("/share/josh/msmarco/queries.dev.small.tsv") as f:
    qid_map = [int(line.split()[0]) for line in f.readlines()]


with open(result_file_name, "w") as f:
    for qid_index in range(len(qid_map)):
        pids = []
        for rank in range(0, 100):
            qid = qid_map[qid_index]
            pid = -1
            while pid == -1 or pid in pids:
                pid = random.randrange(8841823)
            pids.append(pid)
            f.write(f"{qid}\t{pid}\t{rank + 1}\n")

metrics = compute_metrics_from_files(
    "/share/josh/msmarco/qrels.dev.small.tsv", result_file_name
)
print("#####################")
for metric in sorted(metrics):
    print("{}: {}".format(metric, metrics[metric]))
print("#####################", flush=True)
