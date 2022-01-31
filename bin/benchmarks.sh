payload='{
    "text": "Hello, world."
}'

# Send Slack Notification
# Webhook URL for benchmarks channel
URL="https://hooks.slack.com/services/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"
# curl -X POST -H 'Content-type: application/json' \
# --data "$payload" $URL
curl -d @flash_recall_imagenet.png $URL