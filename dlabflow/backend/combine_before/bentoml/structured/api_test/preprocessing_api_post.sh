curl -X 'POST' \
  'http://10.40.217.236:4123/preprocessing/post' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "projectId": "f3a1b2c4-5d6e-4f7a-8b9c-0d1e2f3a4b5c",
  "versionId": "a7c8d9e0-1f2b-3c4d-5e6f-7a8b9c0d1e2f",
  "dataPath": [
    "eda-test/f901eb38-1c1b-4b10-9706-0e1397ac494e/a9f2e8c4-1b6d-4f3b-9e74-2d8c6a1f5b90/titanic.csv"
  ],
  "ratioOfTraining": 80,
  "ratioOfValidation": 10,
  "ratioOfTesting": 10,
  "versionCreateStructuredTabularClassification": {
    "commonEdaColumns":[
      {"columnName": "Survived", "columnType": "NUMERIC"},
      {"columnName": "Pclass", "columnType": "NUMERIC"},
      {"columnName": "Sex", "columnType": "STRING"},
      {"columnName": "Age", "columnType": "NUMERIC"},
      {"columnName": "SibSp", "columnType": "NUMERIC"},
      {"columnName": "Parch", "columnType": "NUMERIC"},
      {"columnName": "Fare", "columnType": "NUMERIC"},
      {"columnName": "Cabin", "columnType": "STRING"},
      {"columnName": "Embarked", "columnType": "STRING"}
    ],
    "numericalTransformations": [
      {"columnName": "Sex", "numericalTransformation": "ONE_HOT_ENCODING"},
      {"columnName": "Cabin", "numericalTransformation": "LABEL_ENCODING"},
      {"columnName": "Embarked", "numericalTransformation": "LABEL_ENCODING"}
    ],
    "missingValueHandlings": [
      {"columnName": "Age", "missingValueHandling": "MEDIAN_IMPUTATION"},
      {"columnName": "Cabin", "missingValueHandling": "MEAN_IMPUTATION"},
      {"columnName": "Cabin", "missingValueHandling": "MEAN_IMPUTATION"}
    ],
    "outlierHandlings": [
      {"columnName": "Age", "outlierHandling": "IQR_REMOVAL"},
      {"columnName": "SibSp", "outlierHandling": "IQR_REMOVAL"},
      {"columnName": "Parch", "outlierHandling": "IQR_REMOVAL"},
      {"columnName": "Fare", "outlierHandling": "IQR_REMOVAL"}
    ]
  }
}'
