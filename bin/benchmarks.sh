# This script runs our benchmarks and logs the results to our MlFlow server.

BASEDIR=$(dirname "$0")
BENCHMARKING_FOLDER=$BASEDIR/../benchmarking

# --------------- Mag Search ---------------
if [ "$RUN_MAGSEARCH" != "n" ]
then
  # Image net embedding search
  # --read_in_entire_dataset \
  IMAGENET_FOLDER=$BENCHMARKING_FOLDER/magsearch/imagenet
  mkdir -p $IMAGENET_FOLDER
  python3 $BASEDIR/../flash/benchmarks/image_search.py \
    /Users/josh/IndexChunks \
      > $IMAGENET_FOLDER/stdout 2> $IMAGENET_FOLDER/stderr
fi

# --------------- Bolt ---------------
if [ "$RUN_BOLT" != "n" ]
then
  # Amazon 670k extreme classification
  AMAZON_FOLDER=$BENCHMARKING_FOLDER/bolt/amazon670k
  mkdir -p $AMAZON_FOLDER
  python3 $BASEDIR/../bolt/benchmarks/amazon670k.py \
    > $AMAZON_FOLDER/stdout 2> $AMAZON_FOLDER/stderr
fi

# Send Slack Notification
URL="https://hooks.slack.com/services/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"
curl -X POST -H 'Content-type: application/json' -d @$BENCHMARKING_FOLDER/request.json $URL
