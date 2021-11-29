import runner
import subprocess


args = ["python3", "bolt/benchmarks/runner.py", "<dataset>", "--hashes_per_table", "<K>", "--num_tables", "<L>", "--sparsity", "<sparsity>", "--runs", "<num_runs>", "--epochs", "<num_epochs>"]
f = open('output_mnist_sh.txt', 'a+')
datasets = ["mnist_sparse_hidden"]
Ks = [7]
Ls = [32, 64, 128]
sparsities = [0.005, 0.01, 0.05, 0.1]
args[10] = "3" # num_runs
args[12] = "10" # num_epochs
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
