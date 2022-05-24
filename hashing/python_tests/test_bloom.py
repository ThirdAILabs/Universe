import math

# Ideally, the bloom filter uses 1.44log_2(1/epsilon) bits per element, where
# epsilon is the false positive rate.

# Get an example to add to the bloom filter
def get_good_i(i, length):
    return [i * j for j in range(length)]


# Get an example that was not added to the bloom filter
def get_bad_i(i, length):
    return [i * j + (j == i % length) for j in range(length)]


def test_bf():
    from thirdai.hashing import BloomFilter
    from time import time

    num_elements = 2**20
    input_dim = 64
    times = []

    # epsilon is the anticipated false positive rate
    for epsilon in [0.1, 0.01, 0.001, 0.0001]:
        start = time()
        # See https://en.wikipedia.org/wiki/Bloom_filter for theory about where
        # the bits_per_element and num_hashes values come from
        bits_per_element = 1.44 * math.log2(1 / epsilon)
        num_hashes = -math.log2(epsilon)

        bf = BloomFilter(
            num_hashes=int(num_hashes + 1),
            requested_total_num_bits=int(bits_per_element * num_elements),
            input_dim=input_dim,
        )

        for i in range(0, num_elements):
            bf.add(get_good_i(i, input_dim))

        for i in range(0, num_elements):
            assert bf.is_present(get_good_i(i, input_dim))

        error_rate = 0
        for i in range(0, num_elements):
            error_rate += bf.is_present(get_bad_i(i, input_dim))
        error_rate /= num_elements

        assert error_rate < epsilon
        print(error_rate, epsilon)
        times.append(time() - start)

    return times


if __name__ == "__main__":
    print(test_bf())
