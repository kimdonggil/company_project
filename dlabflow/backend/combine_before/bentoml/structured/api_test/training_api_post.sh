curl -X 'POST' \
  'http://10.40.217.236:4123/training/post' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "projectId": "f3a1b2c4-5d6e-4f7a-8b9c-0d1e2f3a4b5c",
  "versionId": "a7c8d9e0-1f2b-3c4d-5e6f-7a8b9c0d1e2f",
  "structuredTabularClassificationTrainingRequestBody": {
    "dependentVariable": {
      "columnName": "Survived"
    },
    "independentVariables": [
      {
        "columnName": "Pclass"
      },
      {
        "columnName": "Age"
      },
      {
        "columnName": "SibSp"
      },
      {
        "columnName": "Parch"
      },
      {
        "columnName": "Fare"
      },
      {
        "columnName": "Cabin"
      },
      {
        "columnName": "Sex"
      }
    ]
  },
  "algorithm": "gradient_boosting",
  "tuning": true,
  "advancedSettingForClassification": {
    "earlyStop": 6,
    "optimizationAlgorithm": "optuna",
    "optimizationMetric": "auc",
    "optimizationIterations": 8,
    "learningRateScheduler": true,
    "minimumLearningRate": 0.05,
    "maximumLearningRate": 0.2
  }
}'
