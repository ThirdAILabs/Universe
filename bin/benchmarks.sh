BASEDIR=$(dirname "$0")
BENCHMARKING_FOLDER=$BASEDIR/../benchmarking

RUN_MAGSEARCH=y

# --------------- Mag Search ---------------
if [ "$RUN_MAGSEARCH" == "y" ]
then
  # Image net embedding search
  # /media/scratch/ImageNetDemo/IndexFiles $IMAGENET_FOLDER \
  IMAGENET_FOLDER=$BENCHMARKING_FOLDER/MagSearch/ImageNet
  mkdir -p $IMAGENET_FOLDER
  python3 $BASEDIR/../flash/benchmarks/image_search.py \
    $IMAGENET_FOLDER /Users/josh/IndexChunks --read_in_entire_dataset \
    > $IMAGENET_FOLDER/stdout 2> $IMAGENET_FOLDER/stderr
fi

# --------------- Bolt ---------------
if [ "$RUN_BOLT" == "y" ]
then
  BOLT_FOLDER=$BENCHMARKING_FOLDER/Bolt
  mkdir $BOLT_FOLDER
  # Amazon 670k extreme classification
  AMAZON_FOLDER=$BOLT_FOLDER/Amazon670k
  mkdir $AMAZON_FOLDER
  python3 $BASEDIR/../bolt/benchmarks/amazon670k.py \
    $AMAZON_FOLDER \
    > $AMAZON_FOLDER/stdout 2> $AMAZON_FOLDER/stderr
fi

# Aggregate request into benchmarking/request.json
python3 $BASEDIR/../benchmarking/aggregate_request.py $BENCHMARKING_FOLDER

# Send Slack Notification
URL="https://hooks.slack.com/services/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"
curl -X POST -H 'Content-type: application/json' -d @$BASEDIR/../request.json $URL
