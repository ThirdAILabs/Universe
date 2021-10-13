#!/bin/bash

BASEDIR=$(dirname "$0")
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
UNIT_TESTS=$(ctest -A | tail -3)
cd -

# Empty string accounts for scheduled workflow having no default values
if [ "$RUN_BOLT" == "y" ] || [ "$RUN_BOLT" == "" ]
then
	DATE=$(date '+%Y-%m-%d')
	target=$BASEDIR/../../logs/$DATE
	mkdir -p $BASEDIR/../../logs/
	mkdir -p $target
	NOW=$(date +"%T")
	LOGFILE="../$target/$NOW.txt"
	
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

	while read line ; do
		echo "$line" | grep -q "Epoch: 1"
		if [ $? = 0 ]
		then
			BOLT_MSG+=$(head -n-2 $LOGFILE)
			
			# Kill job and signal an error if accuracy after the third epoch is less than 0.3.
			ACCURACY=$(echo $(tail -5 $LOGFILE | head -1) | grep -Eo '[+-]?[0-9]+([.][0-9]+)')
			PASS_ACCURACY_CHECK=$(echo "$ACCURACY > 0.3" | bc)
			if [[ "$PASS_ACCURACY_CHECK" -eq 0 ]]
			then
				kill $BOLT_PID
				BOLT_MSG+="\nERROR: Accuracy ($ACCURACY) too low after 3 epochs.\n"
				break
			fi

			# Kill job and signal an error if training time after the third epoch is longer than 400 seconds.
			EPOCH_TRAIN_TIME=$(echo $(tail -7 log.txt | head -1) | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | tail -1)
			if [[ "$EPOCH_TRAIN_TIME" -gt 400 ]]
			then
				kill $BOLT_PID
				BOLT_MSG+="\nERROR: Epoch training time ($EPOCH_TRAIN_TIME) longer than expected.\n"
				break
			fi
		fi

		# Kill job and notify an error if total elapsed time of all epochs is greater than 10000 seconds.
		END_TIME=$(date +"%s")
		ELAPSED_TIME=$(($END_TIME - $START_TIME))
		if [ $ELAPSED_TIME -gt 10000 ]
		then
			kill $BOLT_PID
			BOLT_MSG+="ERROR: Timeout exceeded (>10000 seconds)\n"
			break
		fi
	done < <( tail -fn0 $LOGFILE )
	END_TIME=$(date +"%s")
	BOLT_MSG+="Total elapsed time: $(($END_TIME - $START_TIME)) seconds.\n"
	BOLT_MSG+="Full logs can be found in: /home/cicd/logs/$DATE/$NOW.txt\n"
else
	BOLT_MSG="Skipped BOLT benchmarks"
fi
echo -e $BOLT_MSG

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

# Send Slack Notification
# Webhook URL for benchmarks channel
URL="https://hooks.slack.com/services/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"
curl -X POST -H 'Content-type: application/json' \
--data "$payload" $URL

echo ""
# TODO(alan): Add FLASH benchmarks
if [ "$RUN_FLASH" == "y" ] || [ "$RUN_FLASH" == "" ]
then
    echo "Running FLASH benchmarks...logging results into $LOGFILE..."
else
    echo "Skipped FLASH benchmarks"
fi