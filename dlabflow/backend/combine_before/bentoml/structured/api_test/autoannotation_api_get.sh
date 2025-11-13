curl -X POST "http://10.40.217.236:4123/autoannotation/get" \
     -H "Content-Type: application/json" \
     -d '{
           "id": 111
         }' | jq
