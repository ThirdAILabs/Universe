from thirdai import bolt_v2


def bolt_v2_e5_model():
    input_dim = 50000
    text_input1 = bolt_v2.nn.Input(dim=input_dim)
    text_input2 = bolt_v2.nn.Input(dim=input_dim)

    hidden_dim = 2000
    hidden_op = bolt_v2.nn.FullyConnected(
        dim=hidden_dim, sparsity=1.0, input_dim=input_dim, activation="relu"
    )

    text_hidden1 = hidden_op(text_input1)
    text_hidden2 = hidden_op(text_input2)

    output_dim = 50000
    output_op = bolt_v2.nn.FullyConnected(
        dim=output_dim, sparsity=0.01, input_dim=hidden_dim, activation="sigmoid"
    )

    text_output1 = output_op(text_hidden1)
    text_output2 = output_op(text_hidden2)

    labels = bolt_v2.nn.Input(dim=1)

    contrastive_loss = bolt_v2.nn.losses.EuclideanContrastive(
        output_1=text_output1,
        output_2=text_output2,
        labels=labels,
        dissimilar_cutoff_distance=dissimilar_cutoff_distance,
    )

    contrastive_model = bolt_v2.nn.Model(
        inputs=[text_input1, text_input2],
        outputs=[text_output1, text_output2],
        losses=[contrastive_loss],
    )



from thirdai import bolt

def make_model(queries, responses, args):
    model  = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(tokenizer=),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=20,
        integer_target=True,
        options={
            "extreme_classification": True,
            "extreme_num_hashes": 16,
            "use_tanh": True,
            "hidden_bias": True,
            "output_bias": True,
            "extreme_output_dim": 4000,
            "embedding_dimension": 100,
            "rlhf": True,
            "dissimilar_cutoff_distance": 1.0,
        },
    )
    model.train_contrastive("queries.csv", "responses.csv")