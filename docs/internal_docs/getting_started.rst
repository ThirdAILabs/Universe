Getting Started
===============


Training an MNIST model
+++++++++++++++++++++++

This example walks you through an example run of bolt to train on the MNIST
dataset - the dataloading, defining a neural network with sparsity enabled
using bolt and training using the bolt framework.


Prepare data
------------

The following downloads the data into workspace path and stores it for later
use.

..  code-block:: python

    workspace_path = <...>

    mnist_path = os.path.join(workspace_path, "mnist")
    mnist_test_path = os.path.join(workspace_path, "mnist.t")

    mnist_archive_path = os.path.join(workspace_path, "mnist.bz2")
    mnist_test_archive_path = os.path.join(workspace_path, "mnist.t.bz2")

    mnist_base_url = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
    )
    mnist_archive_url = f"{mnist_base_url}/mnist.bz2"
    mnist_test_archive_url = f"{mnist_base_url}/mnist.t.bz2"

    if not os.path.exists(mnist_path):
        os.system(f"curl {mnist_archive_url} --output {mnist_archive_path}")
        os.system(f"bzip2 -d {mnist_archive_path}")

    if not os.path.exists(mnist_test_path):
        os.system(f"curl {mnist_test_archive_url} --output {mnist_test_archive_path}")
        os.system(f"bzip2 -d {mnist_test_archive_path}")

Next, we preprocess data into a bolt compatible format. Bolt accepts data in
several formats - CSV, TSV, Text. The following example uses an SVM compatible
format.

.. code-block:: python

   batch_size = 250
   train_x, train_y = dataset.load_bolt_svm_dataset(mnist_path, batch_size)
   test_x, test_y = dataset.load_bolt_svm_dataset(mnist_test_path, batch_size)
   mnist_dataset = {
       "train_data": train_x,
       "train_labels": train_y,
       "test_data": test_x,
       "test_labels": test_y,
   }



Define bolt model
-----------------


.. code-block:: python

   input_layer = bolt.nn.Input(dim=784)
   hidden_layer = bolt.nn.FullyConnected(dim=256, sparsity=0.1, activation="relu")(
       input_layer
   )
   output_layer = bolt.nn.FullyConnected(dim=10, activation="softmax")(hidden_layer)

   model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

   model.compile(loss=bolt.CategoricalCrossEntropyLoss())



Training
--------

The following section illustrates defining a :obj:`TrainConfig
<thirdai.bolt.TrainConfig>` to spawn the bolt training pipeline.

.. code-block:: python

   tracked_metrics = ["mean_squared_error"]

   eval_config = bolt.EvalConfig().with_metrics(tracked_metrics)
   train_config = (
       bolt.TrainConfig(learning_rate=0.001, epochs=10)
       .with_metrics(tracked_metrics)
   )


   train_metrics = model.train(
       train_data=mnist_dataset["train_data"],
       train_labels=mnist_dataset["train_labels"],
       train_config=train_config,
   )




Testing
-------

The trained model can be used to test as illustrated below:

.. code-block:: python

    test_metrics = model.evaluate(
        test_data=mnist_dataset["test_data"],
        test_labels=mnist_dataset["test_labels"],
        eval_config=eval_config,
    )



Further training options
++++++++++++++++++++++++

Enhancements to the training pipeline is achieved by modifying
:obj:`TrainConfig <thirdai.bolt.TrainConfig>`. In this section, we
demonstrate a few common use-cases. For understanding full capabilities, refer
to the full API documentation.

Logging
-------

The default thirdai training procedure provides minimal information, indicating
progress and only epoch level metrics. Inorder to have much rich information
about what's going on underneath and batch level metrics real-time, use the
logging backend.

See :obj:`thirdai.logging <thirdai.logging>` API documentation for more
details. Logging granularity during training can be controlled by the following
modifications to :obj:`TrainConfig <thirdai.bolt.TrainConfig>`.

.. code-block:: python

   train_config = (
       bolt.TrainConfig(learning_rate=0.001, epochs=10)
       .with_metrics(metrics)
       .with_log_loss_frequency(32)
   )


It's helpful to silence the progress-bar sometimes and use the logging backend
for information. For this, use the ``.silence()`` option.

.. code-block:: python

   train_config = (
       bolt.TrainConfig(learning_rate=0.001, epochs=10)
       .with_metrics(metrics)
       .with_log_loss_frequency(32)
       .silence()
   )

Validation
----------

The following code demonstrates adding a validation-set to the
:obj:`TrainConfig <thirdai.bolt.TrainConfig>` from the MNIST example, to achieve
validation at specified intervals of updates during training. 

.. code-block:: python

   eval_config = bolt.EvalConfig().with_metrics(tracked_metrics)

   train_config = (
       bolt.TrainConfig(learning_rate=0.001, epochs=10)
       .with_metrics(metrics)
       .with_validation(
           [mnist_dataset["test_data"]],
           mnist_dataset["test_labels"],
           eval_config,
           validation_frequency=32,
           save_best_per_metric="mean_squared_error",
       )
   )

Use logging above to see real-time updates on validation metrics.

Saving models
-------------

Inorder to save-models at defined intervals of updates, use the following
additions, making use of the :meth:`.with_save_parameters(...)
<thirdai.bolt.TrainConfig.with_save_parameters>` option:

.. code-block:: python

   train_config = (
       bolt.TrainConfig(learning_rate=0.001, epochs=10)
       .with_metrics(metrics)
       .with_save_parameters(save_prefix="model", save_frequency=32)
   )


Keyboard Interrupts
-------------------

``thirdai`` is made efficient by C++, and made more user-friendly by exposing
the higher-level functions in Python. Due to pybind11/Python behaviour, keyboard-interrupt
is held by the interpreter until the C++ code returns the GIL to Python. The
consequence from a user-perspective is that Ctrl-C is not registered as quickly
as a normal user would expect. Since our training can go long (in C++), it
might take a while before the vanilla keyboard-interrupt mechanisms kick in.

In the current state of things, to enable mid-training keyboard-interrupt, a
callback can be passed to the ``TrainConfig``, as demonstrated below:

.. code-block:: python


   train_config = (
       bolt.TrainConfig(learning_rate=0.001, epochs=10)
       .with_metrics(metrics)
       .with_callbacks([bolt.callbacks.KeyboardInterrupt()])
   )

