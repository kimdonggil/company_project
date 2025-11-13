lsof -ti :4123 | xargs kill -9
rm -rf /mnt/dlabflow/structured/backend/bentoml/bentoml.log
rm -rf /mnt/dlabflow/structured/backend/bentoml/__pycache__
rm -rf /mnt/dlabflow/structured/backend/bentoml/*_pipelines.zip

