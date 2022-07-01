from thirdai import bolt
import pytest
import textwrap


@pytest.mark.unit
def test_simple_bolt_dag_summary():
    input_layer_1 = bolt.graph.Input(dim=10)

    full_layer_1 = bolt.graph.FullyConnected(
        dim=10,
        activation="relu",
    )(input_layer_1)

    full_layer_2 = bolt.graph.FullyConnected(
        dim=10,
        sparsity=0.01,
        activation="relu",
    )(full_layer_1)

    concat_layer = bolt.graph.Concatenate()([full_layer_1, full_layer_2])

    full_layer_3 = bolt.graph.FullyConnected(
        dim=100,
        activation="relu",
    )(concat_layer)

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(full_layer_3)

    model = bolt.graph.Model(inputs=[input_layer_1], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss(), print_when_done=False)

    normal_summary = model.summary(detailed=False, print=False)
    detailed_summary = model.summary(detailed=True, print=False)

    expected_detailed = """
      ======================= Bolt Network =======================
      input1 (Input) : dim=10
      input1 -> full1 (FullyConnected): dim=10, sparsity=1, act_func=ReLU (hashes_per_table=0, num_tables=0, range_pow=0, resevoir_size=0, hash_function=DWTA)
      full1 -> full2 (FullyConnected): dim=10, sparsity=0.01, act_func=ReLU (hashes_per_table=5, num_tables=328, range_pow=15, resevoir_size=4, hash_function=DWTA)
      (full1, full2) -> concat1 (Concatenate)
      concat1 -> full3 (FullyConnected): dim=100, sparsity=1, act_func=ReLU (hashes_per_table=0, num_tables=0, range_pow=0, resevoir_size=0, hash_function=DWTA)
      full3 -> full4 (FullyConnected): dim=10, sparsity=1, act_func=Softmax (hashes_per_table=0, num_tables=0, range_pow=0, resevoir_size=0, hash_function=DWTA)
      ============================================================
    """

    expected_normal = """
      ======================= Bolt Network =======================
      input1 (Input) : dim=10
      input1 -> full1 (FullyConnected): dim=10, sparsity=1, act_func=ReLU
      full1 -> full2 (FullyConnected): dim=10, sparsity=0.01, act_func=ReLU
      (full1, full2) -> concat1 (Concatenate)
      concat1 -> full3 (FullyConnected): dim=100, sparsity=1, act_func=ReLU
      full3 -> full4 (FullyConnected): dim=10, sparsity=1, act_func=Softmax
      ============================================================
    """

    assert normal_summary.strip() == textwrap.dedent(expected_normal).strip()
    assert detailed_summary.strip() == textwrap.dedent(expected_detailed).strip()
