with open("benchmarking/magsearch/imagenet/stdout") as f:
    lines = f.readlines()
    i = 0
    results = []
    while i < len(lines) - 3:
        words = lines[i].split()
        res_size, hashes_per_table, num_tables = [
            int(words[5]),
            int(words[7]),
            int(words[11]),
        ]
        i += 1
        indexing_time = float(lines[i].split(":")[1][:-1])
        i += 1
        querying_time = float(lines[i].split(":")[1])
        i += 1
        recall = float(lines[i].split(" = ")[1])
        i += 1
        results.append(
            (
                res_size,
                hashes_per_table,
                num_tables,
                indexing_time,
                querying_time,
                recall,
            )
        )

time_recall = [(result[4], result[5], result) for result in results]


def pareto(values):
    values.sort()
    results = []
    for v in values:
        if len(results) == 0 or results[-1][1] < v[1]:
            results.append(v)
    return results


time_recall_to_plot = pareto(time_recall)
print(time_recall_to_plot)


import matplotlib.pyplot as plt
import math


xs = [p[1] for p in time_recall_to_plot]
ys = [math.log10(10000 / p[0]) for p in time_recall_to_plot]

fig, ax = plt.subplots()
ax.scatter(xs, ys)

for i in range(len(time_recall_to_plot)):
    ax.annotate(time_recall_to_plot[i][2][:3], (xs[i], ys[i]))

titlefontsize = 22
axisfontsize = 18
labelfontsize = 12
ls = "--"
plt.xlabel("Recall (R10@100)", fontsize=axisfontsize)
plt.ylabel("Queries per second (log 10)", fontsize=axisfontsize)
plt.title("Flash Recall on ImageNet Vgg Embeddings", fontsize=titlefontsize)
plt.show()
