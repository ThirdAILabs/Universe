from thirdai import bolt
import pytest
import textwrap


@pytest.mark.unit
def test_simple_bolt_dag_summary():
    input_layer_1 = bolt.graph.Input(dim=10)

    fc_layer_1 = bolt.graph.FullyConnected(
        dim=10,
        activation="relu",
    )(input_layer_1)

    fc_layer_2 = bolt.graph.FullyConnected(
        dim=10,
        sparsity=0.01,
        activation="relu",
    )(fc_layer_1)

    concat_layer = bolt.graph.Concatenate()([fc_layer_1, fc_layer_2])

    fc_layer_3 = bolt.graph.FullyConnected(
        dim=100,
        activation="relu",
    )(concat_layer)

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(fc_layer_3)

    model = bolt.graph.Model(inputs=[input_layer_1], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss(), print_when_done=False)

    normal_summary = model.summary(detailed=False, print=False)
    detailed_summary = model.summary(detailed=True, print=False)

    expected_normal = """
      ======================= Bolt Model =======================
      input_1 (Input) : dim=10
      input_1 -> fc_1 (FullyConnected): dim=10, sparsity=1, act_func=ReLU
      fc_1 -> fc_2 (FullyConnected): dim=10, sparsity=0.01, act_func=ReLU
      (fc_1, fc_2) -> concat_1 (Concatenate)
      concat_1 -> fc_3 (FullyConnected): dim=100, sparsity=1, act_func=ReLU
      fc_3 -> fc_4 (FullyConnected): dim=10, sparsity=1, act_func=Softmax
      ============================================================
    """

    expected_detailed = """
      ======================= Bolt Model =======================
      input_1 (Input) : dim=10
      input_1 -> fc_1 (FullyConnected): dim=10, sparsity=1, act_func=ReLU
      fc_1 -> fc_2 (FullyConnected): dim=10, sparsity=0.01, act_func=ReLU (hash_function=DWTA, num_tables=328, range=32768, reservoir_size=4)
      (fc_1, fc_2) -> concat_1 (Concatenate)
      concat_1 -> fc_3 (FullyConnected): dim=100, sparsity=1, act_func=ReLU
      fc_3 -> fc_4 (FullyConnected): dim=10, sparsity=1, act_func=Softmax
      ============================================================
    """

    assert normal_summary.strip() == textwrap.dedent(expected_normal).strip()
    assert detailed_summary.strip() == textwrap.dedent(expected_detailed).strip()
