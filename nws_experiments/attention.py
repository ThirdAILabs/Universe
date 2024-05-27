from thirdai.bolt import RACE, SRP
import numpy as np
import time
import math

def normalize(a):
        l2_norms = np.linalg.norm(a, axis=1, keepdims=True)
        return a / l2_norms

def create_qkv(num_clusters, samples_per_cluster, num_columns, noise):
    # Step 1: Generate a random 2D array with positive values
    random_array = np.random.randn(num_clusters, num_columns)  # Generates values between 0 and 1
    q_noisy_arrays = []
    k_noisy_arrays = []
    v_noisy_arrays = []
    for _ in range(samples_per_cluster):
        q_noisy_arrays.append(random_array + noise * np.random.rand(num_clusters, num_columns))
        k_noisy_arrays.append(random_array + noise * np.random.rand(num_clusters, num_columns))
        v_noisy_arrays.append(random_array + noise * np.random.rand(num_clusters, num_columns))
    q = np.concatenate(q_noisy_arrays)
    k = np.concatenate(k_noisy_arrays)
    v = np.concatenate(v_noisy_arrays)
    # Step 2: Normalize each row to have an L2 norm of 1
    return normalize(q), normalize(k), normalize(v)
    # return normalize(q), normalize(k), np.identity(num_clusters * samples_per_cluster)


def attention(queries, keys, values):
    exps = np.exp(np.dot(queries, np.transpose(keys)))
    div = np.sum(exps, axis=1)
    exps /= div.reshape(-1, 1)
    return np.dot(exps, values)


def what_race_wants_to_be(queries, keys, values, mean_coeff, coeff_pow_pairs):
    results = []
    results_bottom = []
    for query in queries:
        results.append(np.zeros((values.shape[-1],) , np.float32))
        results_bottom.append([0.0])
        for key, value in zip(keys, values):
            theta = np.arccos(np.dot(query, key))
            mult = mean_coeff
            for coeff, power in coeff_pow_pairs:
                mult += coeff * ((1 - (theta / math.pi)) ** power)

            results[-1] += mult * value
            results_bottom[-1] += mult
    return np.array(results) / np.array(results_bottom)

def race_attention(queries, keys, values, race_rows, mean_coeff, coeff_pow_pairs):
    final_top = np.zeros((len(queries), values.shape[-1]), np.float32)
    final_bottom = np.zeros((len(queries), 1), np.float32)
    for coeff, power in coeff_pow_pairs:
        srp = SRP(
            input_dim=keys.shape[-1],
            rows=race_rows,
            hashes_per_row=power,
        )
        top = RACE(srp, values.shape[-1])
        bottom = RACE(srp, 1)
        for key, val in zip(keys, values):
            top.update(key, val)
            bottom.update(key, [1])
        top_results = np.array([
            top.query(query)
            for query in queries
        ])
        bottom_results = np.array([
            bottom.query(query)
            for query in queries
        ])
        final_top += coeff * top_results
        final_bottom += coeff * bottom_results
    final_top += mean_coeff * np.mean(values, axis=0)
    final_bottom += mean_coeff
    return final_top / final_bottom


def evaluate(nclusters, samples_per_cluster, ncols, race_rows, noise):
    print(f"{nclusters=} {samples_per_cluster} {ncols=} {race_rows}")
    queries, keys, values = create_qkv(nclusters, samples_per_cluster, ncols, noise)
    mean_coeff = 0.35
    coeff_pow_pairs = [(2.8, 2)]
    # What race wants to be
    start_truth = time.time()
    truth = what_race_wants_to_be(queries, keys, values, mean_coeff, coeff_pow_pairs)
    # truth = attention(queries, keys, values)
    end_truth = time.time()
    print(f"What race wants to be finished in {end_truth - start_truth} seconds.")
    # Approx attention
    start_race = time.time()
    approx = race_attention(queries, keys, values, race_rows, mean_coeff, coeff_pow_pairs)
    # approx = what_race_wants_to_be(queries, keys, values, mean_coeff, coeff_pow_pairs)
    end_race = time.time()
    print(f"RACE attention finished in {end_race - start_race} seconds.")
    # Compute error
    avg = np.mean(values, axis=0)
    avg = np.array([avg for _ in queries])
    avg = avg
    print("Avg truth l2", np.mean(np.linalg.norm(truth, axis=1)))
    print("Avg approx l2", np.mean(np.linalg.norm(approx, axis=1)))
    print("Avg avg l2", np.mean(np.linalg.norm(avg, axis=1)))
    print("DOTS", np.mean(np.sum(truth * approx, axis=1)))
    print("DOTS median", np.median(np.sum(truth * approx, axis=1)))
    print("DOTS AVG", np.mean(np.sum(truth * avg, axis=1)))
    print("DOTS median AVG", np.median(np.sum(truth * avg, axis=1)))
    true_l2 = np.linalg.norm(truth, axis=1)
    approx_l2 = np.linalg.norm(approx, axis=1)
    avg_l2 = np.linalg.norm(avg, axis=1)
    print("sim", np.mean(np.sum(truth * approx, axis=1) / true_l2 / approx_l2 * ((true_l2 - np.abs(true_l2 - approx_l2)) / true_l2)))
    print("sim avg", np.mean(np.sum(truth * avg, axis=1) / true_l2 / avg_l2 * ((true_l2 - np.abs(true_l2 - avg_l2)) / true_l2)))
    print("THETA", np.mean(np.arccos(np.sum(truth * approx, axis=1) / np.linalg.norm(truth, axis=1) / np.linalg.norm(approx, axis=1))))
    print("THETA median", np.median(np.arccos(np.sum(truth * approx, axis=1) / np.linalg.norm(truth, axis=1) / np.linalg.norm(approx, axis=1))))
    print("THETA AVG", np.mean(np.arccos(np.sum(truth * avg, axis=1) / np.linalg.norm(truth, axis=1) / np.linalg.norm(avg, axis=1))))
    print("THETA median AVG", np.median(np.arccos(np.sum(truth * avg, axis=1) / np.linalg.norm(truth, axis=1) / np.linalg.norm(avg, axis=1))))
    dif = np.mean(np.linalg.norm(truth - approx, axis=1))
    dif_median = np.median(np.linalg.norm(truth - approx, axis=1))
    print("DIF", dif, dif / np.mean(np.linalg.norm(truth, axis=1)))
    print("DIF median", dif_median, dif_median / np.mean(np.linalg.norm(truth, axis=1)))
    dif_avg = np.mean(np.linalg.norm(truth - avg, axis=1))
    dif_median_avg = np.median(np.linalg.norm(truth - avg, axis=1))
    print("DIF AVG", dif_avg, dif_avg / np.mean(np.linalg.norm(truth, axis=1)))
    print("DIF median AVG", dif_median_avg, dif_median_avg / np.mean(np.linalg.norm(truth, axis=1)))
    print("Mean error:", np.mean(np.abs(approx - truth)))
    print("Mean percentage error:", np.mean(np.abs(approx - truth) / np.abs(truth)) * 100)
    print("Mean percentage error avg:", np.mean(np.abs(avg - truth) / np.abs(truth)) * 100)



evaluate(10, 100, 1000, 50, 0.001)
