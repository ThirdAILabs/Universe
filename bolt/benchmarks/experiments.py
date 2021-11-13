import runner
import subprocess


args = ["python3", "bolt/benchmarks/runner.py", "<dataset>", "--K", "<K>", "--L", "<L>", "--sparsity", "<sparsity>", "--runs", "<num_runs>"]
f = open('output_amzn670k.txt', 'w+')
datasets = ["amzn670"]
Ks = [3,5,7]
Ls = [32, 64, 128]
sparsities = [0.001, 0.005, 0.01]
args[10] = "1"
for dataset in datasets:
    for K in Ks:
        for L in Ls:
            for s in sparsities:
                f.write("K: " + str(K) + "\n")
                f.write("L: " + str(L) + "\n")
                f.write("sparsity: " + str(s) + "\n")
                f.flush()
                args[2] = dataset
                args[4] = str(K)
                args[6] = str(L)
                args[8] = str(s)
                p = subprocess.Popen(args, stdout=subprocess.PIPE)
                f.write(str(p.stdout.readlines()[-2:]))
                f.write("\n----------------------\n")
                f.flush()
