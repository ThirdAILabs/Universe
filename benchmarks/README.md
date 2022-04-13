## FAQ About MlFlow

## How does MlFlow work?
MlFlow is a metadata and artifact store for testing and benchmarking machine
learning models and algorithms. You create a model or algorithm on your 
local or cloud machine, run your normal training or testing process, and 
report all sorts of metadata to a "tracking server". Our tracking server
is currently hosted on AWS Fargate. We also have a currently unused artifact
store on S3, integrated with our tracking server, so eventually the models
or indices we build can be stored for later examination.


## How do I log to MlFlow at ThirdAI?
Logging to MlFlow is simple. To start logging, simply import mlflow_logger.py to your script.
This is stored in Universe/logging. If calling from a script somewhere in 
Universe, current recommended usage is to add Universe/logging to your path
in the script and then import the relevant logging class or method. 
For example usage, see bolt/benchmarks/amazon670k.py or flash/benchmarks/image_search.py. 
For Bolt or another machine learning method, your benchmark code will look something like
```
import sys
sys.path.insert(1, sys.path[0] + "/../../logging/")
from mlflow_logger import ModelLogger

...

with ModelLogger(
    dataset="amazon670k",
    learning_rate=0.01,
    num_hash_tables=10,
    hashes_per_table=5,
    sparsity=0.01,
    algorithm="bolt") as mlflow_logger:

<Code to init model>

mlflow_logger.log_start_training()

for each epoch:
    <train model a single epoch>
    <get model test accuracy>
    mlflow_logger.log_epoch(accuracy)


```
For MagSearch, your code will look something like
```
import sys
sys.path.insert(1, sys.path[0] + "/../../logging/")
from mlflow_logger import log_magsearch_run

...

<build flash index, get indexing time>
<query flash index, get querying time and recall>

mlflow_logger.log_magsearch_run(
    reservoir_size=reservoir_size,
    hashes_per_table=hashes_per_table,
    num_tables=num_tables,
    indexing_time=indexing_time,
    querying_time=querying_time,
    num_queries=10000,
    recall=recall,
    dataset="imagenet_embeddings"
)

```
For new uses of MlFlow not encompassed in these two methods,
it is preferred that you implement a new method in mlflow_logger.py.

## How do I examine MlFlow runs?
Currently, our tracking server is hosted at:
http://deplo-mlflo-15qe25sw8psjr-1d20dd0c302edb1f.elb.us-east-1.amazonaws.com/#/
Simply navigate to that url in your browser to see the intuitive mlflow UI. In 
any experiment you can examine an individual run and find their parameters
and metrics, or select multiple runs and compare them on any parameter or metric 
they share.

We plan to find or add a better UI for generating and saving graphs on top of 
MlFlow. See https://github.com/ThirdAILabs/Universe/issues/220 for progress.