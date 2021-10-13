#!/bin/bash

export BASEDIR=$(dirname "$0")
./$BASEDIR/build.sh

CURRENT_BRANCH=$(git branch --show-current)
REPO_URL="https://github.$(git config remote.origin.url | cut -f2 -d. | tr ':' /)"
BRANCH_URL="$REPO_URL/tree/$CURRENT_BRANCH"
ACTIONS_URL="$REPO_URL/actions/runs/$RUN_ID"
MODEL_NAME=$(grep -m 1 "model name" /proc/cpuinfo | sed -e "s/^.*: //")
NUM_CPUS=$(grep -c ^processor /proc/cpuinfo)
OTHER_MACHINE_INFO=$(lscpu | egrep 'Socket|Thread|Core')
CODE_VERSION+=$(git describe --tag)
cd $BASEDIR/../build/
#UNIT_TESTS=$(ctest -A | tail -3)
cd -

if [ "$RUN_BOLT" == "y" ] | [ "$RUN_BOLT" == "" ]
then
    echo "Running BOLT benchmarks..."
	DATE=$(date '+%Y-%m-%d')
	target=$BASEDIR/../../logs/$DATE
	mkdir -p $BASEDIR/../../logs/
	mkdir -p $target
	NOW=$(date +"%T")
	LOGFILE="$target/$NOW.txt"
	LOGFILE="log.txt"

	START_TIME=$(date +"%s")
	./build/bolt/bolt bolt/configs/amzn_benchmarks.cfg > $LOGFILE &
	BOLT_PID=$!

	tail -fn0 $LOGFILE | \
	while read line ; do
		echo "Checking accuracy and time... $line" | grep "Epoch 3:"
		if [ $? = 0 ]
		then
			ACCURACY=$(echo $(tail -4 $LOGFILE | head -1) | grep -Eo '[+-]?[0-9]+([.][0-9]+)')
			PASS_ACCURACY_CHECK=$(echo "$ACCURACY > 0.3" | bc)
			if [ $PASS_ACCURACY_CHECK -eq 0 ]
			then
				kill $BOLT_PID
				BOLT_MSG+="ERROR: Accuracy ($ACCURACY) too low after 3 epochs."
				break
			fi
		fi

		END_TIME=$(date +"%s")
		ELAPSED_TIME=$(($END_TIME - $START_TIME))
		echo $ELAPSED_TIME
		if [ $ELAPSED_TIME -gt 5 ]
		then
			kill $BOLT_PID
			BOLT_MSG+="ERROR: Timeout exceeded (>10000 seconds)"
			break
		fi
	done
	BOLT_MSG="BOLT epoch logs can be found in: /home/cicd/logs/$DATE/$NOW.txt"
else
	BOLT_MSG="Skipped BOLT benchmarks"
fi
echo $BOLT_MSG

# TODO(alan): Decide if we want to attach full logs (will need to use S3 or similar to upload remote logs)
# curl -F token=* \
#     -F external_id=LOG \
#     -F external_url="file://~/logs/$DATE/$NOW.txt" \
#     -F title=logs \
#     https://slack.com/api/files.remote.add

payload="
{
    \"text\": \"Benchmarks Completed\",
    \"blocks\": [
        {
			\"type\": \"section\",
			\"text\": {
                \"type\": \"mrkdwn\", 
                \"text\": \"------------------ *_Benchmarks for $DATE $NOW _*------------------\n\"
            }
		},
		{
			\"type\": \"section\",
			\"text\": {
                \"type\": \"mrkdwn\", 
                \"text\": \"Branch: <$BRANCH_URL|$CURRENT_BRANCH>\n<$ACTIONS_URL|GitHub Actions Logs>\"
            }
		},
		{
			\"type\": \"header\",
			\"text\": {
				\"type\": \"plain_text\",
				\"text\": \"Machine Information\"
			}
		},
		{
			\"type\": \"section\",
			\"fields\": [
                {
					\"type\": \"mrkdwn\",
					\"text\": \"*Processor:*\n$MODEL_NAME\"
				},
                {
					\"type\": \"mrkdwn\",
					\"text\": \"*CPU(s):*\n$NUM_CPUS\"
				},
			]
		},
		{
			\"type\": \"section\",
			\"fields\": [
				{
					\"type\": \"mrkdwn\",
					\"text\": \"*Other:*\n$OTHER_MACHINE_INFO\"
				},
			]
		},
		{
			\"type\": \"header\",
			\"text\": {
				\"type\": \"plain_text\",
				\"text\": \"Current Version\"
			}
		},
        {
			\"type\": \"section\",
			\"fields\": [
				{
					\"type\": \"mrkdwn\",
					\"text\": \"$CODE_VERSION\"
				}
			]
		},
        {
			\"type\": \"header\",
			\"text\": {
				\"type\": \"plain_text\",
				\"text\": \"Unit Tests\"
			}
		},
        {
            \"type\": \"section\",
			\"text\": {
                \"type\": \"mrkdwn\", 
                \"text\": \"$UNIT_TESTS\"
            }
        },
		{
			\"type\": \"header\",
			\"text\": {
				\"type\": \"plain_text\",
				\"text\": \"Bolt Benchmarks\"
			}
		},
        {
            \"type\": \"section\",
			\"text\": {
                \"type\": \"mrkdwn\", 
                \"text\": \"$BOLT_MSG\"
            }
        },
	]
}"

# ../$BASEDIR/bolt_mnist_test.sh >> $LOGFILE

# Send Slack Notification
# Webhook URL for benchmarks channel
URL="https://hooks.slack.com/services/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"
curl -X POST -H 'Content-type: application/json' \
--data "$payload" $URL

# TODO(alan): Add FLASH benchmarks
if [ "$RUN_FLASH" == "y" ]
then
    echo "Running FLASH benchmarks...logging results into $LOGFILE..."
else
    echo "Skipped FLASH benchmarks"
fi