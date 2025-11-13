curl -X 'POST' \
    'http://10.40.217.236:4123/eda/post' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "realFileId": 12345,
    "filePath": "eda-test/f901eb38-1c1b-4b10-9706-0e1397ac494e/a9f2e8c4-1b6d-4f3b-9e74-2d8c6a1f5b90/titanic.csv"
}'
