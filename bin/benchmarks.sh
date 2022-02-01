BASEDIR=$(dirname "$0")
BENCHMARKING_FOLDER=$BASEDIR/../benchmarking

RUN_MAGSEARCH=y

# --------------- Mag Search ---------------
if [ "$RUN_MAGSEARCH" == "y" ]
then
  # Image net embedding search
  # /media/scratch/ImageNetDemo/IndexFiles $IMAGENET_FOLDER \
  IMAGENET_FOLDER=$BENCHMARKING_FOLDER/magsearch/imagenet
  mkdir -p $IMAGENET_FOLDER
  python3 $BASEDIR/../flash/benchmarks/image_search.py \
    /Users/josh/IndexChunks $IMAGENET_FOLDER \
    > $IMAGENET_FOLDER/stdout 2> $IMAGENET_FOLDER/stderr
fi

# --------------- Bolt ---------------
if [ "$RUN_BOLT" == "y" ]
then
  BOLT_FOLDER=$BENCHMARKING_FOLDER/bolt
  mkdir $BOLT_FOLDER
  # Amazon 670k extreme classification
  AMAZON_FOLDER=$BOLT_FOLDER/amazon670k
  mkdir $AMAZON_FOLDER
  # TODO: ADD HERE
fi

# python3 $BASEDIR/../benchmarking/aggregate_request.py $BENCHMARKING_FOLDER

# python3 $BASEDIR/../benchmarking/send_request_to_slack.py $BENCHMARKING_FOLDER/request.json