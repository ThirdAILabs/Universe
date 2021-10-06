#!/bin/bash

export BASEDIR=$(dirname "$0")
./$BASEDIR/build.sh
./$BASEDIR/get_datasets.sh

export DATE=$(date '+%Y-%m-%d')
target=$BASEDIR/../../logs/$DATE
mkdir -p $BASEDIR/../../logs/
mkdir -p $target
export NOW=$(date +"%T")

# We need: Code version, machine information, run time, accuracy, hash seeds
cd $BASEDIR/../build/
LOGFILE="../$target/$NOW.txt"
echo "Logging into $LOGFILE"

# TODO(Alan, Henry): 
# - Add bolt benchmark tests and several configs.
# - Improve formatting (maybe hide some information)
# - 
MSG=$'*_-------------------- Machine information --------------------_*\n'
MSG+=$(lscpu)
MSG+=$'\n\n\n*_-------------------- Current code version --------------------_*\n'
MSG+=$(git describe --tag)
MSG+=$'\n\n\n*_-------------------- Date --------------------_*\n'
MSG+=$DATE
MSG+=$'\n\n\n*_-------------------- Unit Test Results --------------------_*\n'
MSG+=$(ctest -A)
echo MSG >> $LOGFILE

# JSON_STRING=$( jq -n \
#                   --arg ver "$VERSION" \
#                   --arg tst "$CTEST_OUTPUT" \
#                   --arg info "$MACHINE_INFO" \
#                   '{text: $tst}' )

# # ../$BASEDIR/bolt_mnist_test.sh >> $LOGFILE

# Send Slack Notification
# Webhook URL for benchmarks channel
URL="https://hooks.slack.com/services/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"

# TODO(alan): Learn how to format this json better.
curl -X POST -H 'Content-type: application/json' \
--data "{\"text\": \"$MSG\"}" $URL