Distributed Training with Bolt
==============================

Introduction
------------
The Bolt library provides a distributed training framework for Bolt models. It enables you to train your models efficiently across multiple workers in a data parallel distributed environment. This document will guide you on how to use the Bolt distributed module for training your models.

Initializing a Ray Cluster
--------------------------

To use Bolt for distributed training, you need to initialize a Ray cluster to manage the distributed computations. Ray provides a convenient way to set up and manage the cluster. Here are the steps to connect to a Ray cluster:

1. Import the necessary modules:

.. code-block:: python

    import ray
    from thirdai import bolt_v2 as bolt


2. Initialize Ray:

The `ray.init()` function initializes Ray and sets up the necessary infrastructure for distributed training. By default, it starts a Ray cluster on your local machine using available resources.

.. code-block:: python

    ray.init()


3. Configure Ray resources:

You can configure the resources allocated to Ray workers using the `num_cpus` and `num_gpus` arguments in `ray.init()`. For example:

.. code-block:: python

    ray.init(num_cpus=4, num_gpus=2)


   This configures Ray to use 4 CPUs and 2 GPUs for distributed training.


4. Optional: Additional Ray configuration:

You can provide additional configuration options to `ray.init()` as per your requirements. For example, you can specify the working directory or set environment variables. Here's an example:

.. code-block:: python

    ray.init(
        runtime_env={
            "working_dir": "/path/to/working/directory",
            "env_vars": {"OMP_NUM_THREADS": "4"},
        }
    )

    In the example above, the working directory is set to `/path/to/working/directory`, and the environment variable `OMP_NUM_THREADS` is set to `4`.

By following these steps, you can connect to a Ray cluster to support distributed training with Bolt. Once the cluster is initialized, you can proceed with creating a Bolt trainer and starting the distributed training process.

Please note that the exact configuration and initialization steps may vary depending on your specific use case and cluster setup. It is recommended to refer to the Ray documentation for detailed instructions on cluster initialization and configuration.

Installation
------------
To use the Bolt distributed module, you need to install the ThirdAI library, which includes the Bolt library. You can install it using pip:

.. code-block:: python

    pip3 install thirdai


Import Statements
----------------
To use the distributed Bolt module, you need to import the necessary modules from the ThirdAI library. Here are the import statements you need:

.. code-block:: python
    
    import thirdai.distributed_bolt as dist
    from thirdai import bolt_v2 as bolt


Distributed Training Workflow
----------------------------
The general workflow for distributed training with Bolt consists of the following steps:

1. Define your model: Create a Bolt model that represents your machine learning model architecture.

2. Prepare your data: Prepare your training and validation datasets. Bolt supports various data formats, such as NumPy arrays and Bolt tensors.

3. Define the training loop: Define a training loop function that takes in a configuration and performs the training logic. This function will be executed by each worker in parallel.

4. Initialize the Bolt trainer: Create an instance of the `dist.BoltTrainer` class, passing the necessary arguments such as the training loop function, model, and scaling configuration.

5. Start distributed training: Call the `fit()` method on the Bolt trainer instance to start the distributed training process. This method will automatically distribute the training workload across the available workers.

6. Monitor training progress: You can monitor the training progress by accessing the training history and checkpoints returned by the `fit()` method. You can also use the `validate()` method to evaluate your model's performance on validation data during training.

7. Save and load checkpoints: You can save and load checkpoints during training using the `dist.BoltCheckPoint` class. Checkpoints allow you to resume training from a specific point or perform inference with a trained model.

Example Usage
-------------

Here's an example usage of the Bolt distributed module:

.. code-block:: python

    import thirdai.distributed_bolt as dist
    from thirdai import bolt_v2 as bolt

    def train_loop_per_worker(config):
        # Training logic goes here
        pass

    # Define your model
    model = ...

    # Prepare your data
    train_x, train_y = ...
    test_x, test_y = ...

    # Create a Bolt trainer
    scaling_config = bolt.ScalingConfig(num_workers=4, use_gpu=True)
    trainer = dist.BoltTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={...},
        scaling_config=scaling_config,
    )

    # Start distributed training
    result_checkpoint_and_history = trainer.fit()

    # Perform validation
    model = dist.BoltCheckPoint.get_model(result_checkpoint_and_history.checkpoint)
    trainer = bolt.train.Trainer(model)
    history = trainer.validate(...)

    # Save and load checkpoints
    checkpoint = dist.BoltCheckPoint.from_model(model)
    checkpoint.save("checkpoint.pth")
    loaded_checkpoint = dist.BoltCheckPoint.load("checkpoint.pth")
    loaded_model = dist.BoltCheckPoint.get_model(loaded_checkpoint)


Documentation Reference
-----------------------

For detailed API reference and usage examples, please refer to the Bolt documentation.
