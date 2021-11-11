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
# Run Unit Tests with valgrind to check for memory leaks.
UNIT_TESTS=$(ctest -T memcheck -E "TimeTest" | tail -5)
cd -

# Empty string accounts for scheduled workflow having no default values
if [ "$RUN_BOLT" == "y" ] || [ "$RUN_BOLT" == "" ]
then
    echo "Running BOLT benchmarks..."
	DATE=$(date '+%Y-%m-%d')
	target=$BASEDIR/../../logs/$DATE
	mkdir -p $BASEDIR/../../logs/
	mkdir -p $target
	NOW=$(date +"%T")
	LOGFILE="$target/$NOW.txt"

	# Run bolt benchmark on amzn670k (see python script for configurations).
	START_TIME=$(date +"%s")
	python3 bolt/benchmarks/runner.py amzn670 --enable_checks --epochs 3 --K 5  > $LOGFILE
	if [ $? -eq 1 ]
	then
		BOLT_MSG=$(tail -n 1 $LOGFILE)
		BOLT_MSG+="\n"
		BOLT_MSG+="\n*ERROR*: Terminated run\n"
	else
		python3 bolt/benchmarks/runner.py amzn670 --enable_checks --K 5 > $LOGFILE
		BOLT_MSG=$(tail -n 5 $LOGFILE)
		BOLT_MSG+="\n"
		END_TIME=$(date +"%s")
		BOLT_MSG+="*SUCCESS*: _Total elapsed time: $(($END_TIME - $START_TIME)) seconds._\n"
		BOLT_MSG+="_Full logs can be found in: /home/cicd/logs/$DATE/$NOW.txt_\n"
	fi
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
				\"text\": \"Unit Tests (valgrind)\"
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
    echo "Running FLASH benchmarks..."
else
    echo "Skipped FLASH benchmarks"
fi