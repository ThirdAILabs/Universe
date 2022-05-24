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
        # the bits_per_element and bit_per_table come from
        bits_per_element = 1.44 * math.log2(1 / epsilon)
        num_tables = -math.log2(epsilon)
        bits_per_table = bits_per_element * num_elements / num_tables

        bf = BloomFilter(
            num_tables=int(num_tables + 1),
            table_range_pow=int(math.log2(bits_per_table) + 1),
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
        times.append(time() - start)

    return times


if __name__ == "__main__":
    print(test_bf())
