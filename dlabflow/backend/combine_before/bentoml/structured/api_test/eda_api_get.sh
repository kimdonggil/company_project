curl -X POST "http://10.40.217.236:4123/eda/get" \
     -H "Content-Type: application/json" \
     -d '{"realFileId": 12345}' | jq
