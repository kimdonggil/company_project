import os
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union
import bentoml
from datetime import timedelta
from bentoml.io import JSON
from minio import Minio
import redis
import subprocess
import json
from dotenv import load_dotenv, dotenv_values
import traceback
import logging
import pandas as pd
import base64
from io import StringIO, BytesIO

logger = logging.getLogger('BentoML Log')
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.propagate = False

dotenv_path = '/mnt/dlabflow/structured/config'
load_dotenv(dotenv_path)

RedisHost = os.getenv('RedisHost')
RedisPort = os.getenv('RedisPort')
RedisDB = os.getenv('RedisDB')

MinIODefaultPath = os.getenv('MinIODefaultPath')
MinioBucketUser = os.getenv('MinioBucketUser1')
MinioBucketEDA = os.getenv('MinioBucketEDA')
MinioEndpoint = os.getenv('MinioEndpoint')
MinioAccessKey = os.getenv('MinioAccessKey')
MinioSecretKey = os.getenv('MinioSecretKey')
MinioSecure = os.getenv('MinioSecure', 'False').lower() == 'true'

r = redis.Redis(host=RedisHost, port=RedisPort, db=RedisDB)

class EDAParams(BaseModel):
    realFileId: int = Field(default=12345)
    filePath: str = Field(default='eda-test/f901eb38-1c1b-4b10-9706-0e1397ac494e/a9f2e8c4-1b6d-4f3b-9e74-2d8c6a1f5b90/titanic.csv')
    status: Optional[str] = None
    statusMessage: Optional[str] = None
    edaResult: Optional[str] = None

class CommonEdaColumn(BaseModel):
    columnName: str
    columnType: str

class NumericalTransformation(BaseModel):
    columnName: str
    numericalTransformation: str

class MissingValueHandling(BaseModel):
    columnName: str
    missingValueHandling: str

class OutlierHandling(BaseModel):
    columnName: str
    outlierHandling: str

class VersionCreateStructuredTabularClassification(BaseModel):
    commonEdaColumns: Optional[List[CommonEdaColumn]] = []
    numericalTransformations: Optional[List[NumericalTransformation]] = []
    missingValueHandlings: Optional[List[MissingValueHandling]] = []
    outlierHandlings: Optional[List[OutlierHandling]] = []

class PreprocessingParams(BaseModel):
    projectId: str = Field(default='f56e2ee9-a343-47fc-89bc-af5f86c3a2f7')
    versionId: str = Field(default='6828b249-b614-4070-832b-5aa5b64b5cf2')
    dataPath: List[str] = Field(
        default_factory=lambda: [
            'eda-test/f901eb38-1c1b-4b10-9706-0e1397ac494e/a9f2e8c4-1b6d-4f3b-9e74-2d8c6a1f5b90/titanic.csv',
            'eda-test/f901eb38-1c1b-4b10-9706-0e1397ac494e/a9f2e8c4-1b6d-4f3b-9e74-2d8c6a1f5b90/titanic2.csv'
        ]
    )
    ratioOfTraining: int = Field(default=80)
    ratioOfValidation: int = Field(default=10)
    ratioOfTesting: int = Field(default=10)
    versionCreateStructuredTabularClassification: Optional[VersionCreateStructuredTabularClassification] = None
    status: Optional[str] = None
    statusMessage: Optional[str] = None
    preprocessingResult: Optional[dict] = None
    progress: Optional[int] = None
    trainData: Optional[int] = None
    validationData: Optional[int] = None
    testData: Optional[int] = None

class Variable(BaseModel):
    columnName: str

class StructuredTabularClassificationTrainingRequestBody(BaseModel):
    dependentVariable: Variable
    independentVariables: List[Variable]    

class TrainingParams(BaseModel):
    projectId: str = Field(default='f56e2ee9-a343-47fc-89bc-af5f86c3a2f7')
    versionId: str = Field(default='6828b249-b614-4070-832b-5aa5b64b5cf2')
    structuredTabularClassificationTrainingRequestBody: Optional[StructuredTabularClassificationTrainingRequestBody] = None
    algorithm: str = Field(default='lightgbm')
    tuning: bool = Field(default=True)
    advancedSettingForClassification: Optional[dict] = None
    status: Optional[str] = None
    statusMessage: Optional[str] = None
    trainingResult: Optional[dict] = None
    progress: Optional[int] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    auc: Optional[float] = None    

class InferenceParams(BaseModel):
    projectId: str = Field(default='')
    versionId: str = Field(default='')
    sessionId: str = Field(default='')
    inferenceTabularFile: str = 'titanic_inference_survived.csv'
    status: Optional[str] = None
    statusMessage: Optional[str] = None
    inferenceResult: Optional[dict] = None
    progress: Optional[int] = None

class ClassDefinition(BaseModel):
    className: str
    imageDatasetAnnotationTechnique: str

class AutoannotationParams(BaseModel):
    id: int = Field(default=111)
    projectId: str = Field(default='f56e2ee9-a343-47fc-89bc-af5f86c3a2f7')
    versionId: str = Field(default='6828b249-b614-4070-832b-5aa5b64b5cf2')
    datasetId: str = Field(default='5a14062b-c6c2-42b1-82f2-6ac4fbc68c18')
    autoAlgorithm: str = Field(default='yolo_version_8_small')
    minConfidence: float = Field(default=0.3)
    maxConfidence: float = Field(default=1.0)
    targetImagePaths: List[str] = Field(
        default_factory=lambda: [
            'dlab-v2-prototype-test/dataset/5a14062b-c6c2-42b1-82f2-6ac4fbc68c18/6354.jpg',
            'dlab-v2-prototype-test/dataset/5a14062b-c6c2-42b1-82f2-6ac4fbc68c18/6329.jpg'
        ]
    )
    classDefinitions: Optional[List[ClassDefinition]] = None
    status: Optional[str] = None
    statusMessage: Optional[str] = None
    autoAnnotationResult: Optional[Union[dict, list, str]] = None
    progress: Optional[int] = None

svc = bentoml.Service('example')

@svc.api(input=JSON(pydantic_model=EDAParams), output=JSON(), route='/eda/post')
async def eda_post(params: EDAParams):
    realFileId = params.realFileId
    filePath = params.filePath
    status = params.status
    statusMessage = params.statusMessage
    edaResult = params.edaResult
    key = f"{realFileId}"
    try:
        current_status_data = {'status': 'READY', 'statusMessage': '요청 값이 모두 들어왔습니다.'}
        r.set(key, json.dumps(current_status_data))        
        exit_code = os.system(
            f"python3 /mnt/dlabflow/structured/backend/kubeflow/pipelines/eda.py "
            f"--realFileId={realFileId} --filePath={filePath} --status={status} --statusMessage={statusMessage} --edaResult={edaResult}"
        )
        if exit_code == 0:
            status = 'RUNNING'
            statusMessage = 'EDA 처리 중입니다.'
        else:
            status = 'ERROR'
            statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
            logger.error(f"{statusMessage}")
            raise
    except Exception:
        status = 'ERROR'
        statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
        logger.error(f"{statusMessage}")
        raise
    current_status_data = {'status': status, 'statusMessage': statusMessage}
    r.set(key, json.dumps(current_status_data))
    return {
        'realFileId': realFileId,
        'status': status,
        'statusMessage': statusMessage
    }

@svc.api(input=JSON(pydantic_model=EDAParams), output=JSON(), route='/eda/get')
def eda_get(params: EDAParams):
    key = f"{params.realFileId}"
    error_message = 'BentoML에서 EDA 처리 중 오류가 발생하였습니다.'
    current_status_json = r.get(key)
    if current_status_json is None:
        current_status_data = {'status': None, 'statusMessage': '들어 온 값이 없습니다.'}
    else:
        current_status_data = json.loads(current_status_json)
    if params.status:
        current_status_data['status'] = params.status
        if params.statusMessage:
            current_status_data['statusMessage'] = params.statusMessage
        else:
            if params.status == 'READY':
                current_status_data['statusMessage'] = '요청 값이 모두 들어왔습니다.'
            elif params.status == 'RUNNING':
                current_status_data['statusMessage'] = 'EDA 처리 중입니다.'
            elif params.status == 'FINISH':
                current_status_data['statusMessage'] = 'EDA 처리가 완료되었습니다. '
            elif params.status == 'ERROR':
                current_status_data['statusMessage'] = error_message
    r.set(key, json.dumps(current_status_data))

    eda_result = None
    if current_status_data.get('status') == 'FINISH':
        base_paths = f"/{MinIODefaultPath}/{MinioBucketUser}/{params.realFileId}/{MinioBucketEDA}"
        try:
            dataset_stats = pd.read_csv(f"{base_paths}/dataset_stats.csv")
            missing_stats = pd.read_csv(f"{base_paths}/missing_stats.csv")
            variable_types = pd.read_csv(f"{base_paths}/variable_types.csv")

            dataset_stats_dict = {
                k: int(v) if k in ['numberOfColumns', 'numberOfRows', 'numberOfDuplicateRows'] else (v.item() if hasattr(v, 'item') else v)
                for k, v in zip(dataset_stats['dataStatistics'], dataset_stats['dataStatisticsValues'])
            }

            columns_list = [
                {
                    'columnName': row['columnName'],
                    'columnType': row['type'],
                    'missingValue': row['missingValue'].item() if hasattr(row['missingValue'], 'item') else row['missingValue']
                }
                for _, row in missing_stats.iterrows()
            ]

            number_of_column_type_list = [
                {
                    'columnType': k,
                    'numberOfType': v.item() if hasattr(v, 'item') else v
                }
                for k, v in zip(variable_types['columnTypes'], variable_types['columnTypesValues'])
            ]

            eda_result = {
                'dataStatistics': dataset_stats_dict,
                'columns': columns_list,
                'columnTypes': number_of_column_type_list
            }
        except Exception:
            logger.error(f"{error_message}")
            current_status_data['status'] = 'ERROR'
            current_status_data['statusMessage'] = error_message
            eda_result = None

    response_json = {
        'realFileId': params.realFileId,
        'dataPath': getattr(params, 'dataPath', None),
        'status': current_status_data.get('status'),
        'statusMessage': current_status_data.get('statusMessage'),
        'edaResult': eda_result
    }

    logger.info(f"{json.dumps(response_json, ensure_ascii=False, indent=4)}")
    return response_json

@svc.api(input=JSON(pydantic_model=PreprocessingParams), output=JSON(), route='/preprocessing/post')
async def preprocessing_post(params: PreprocessingParams):
    projectId = params.projectId
    versionId = params.versionId
    file_paths = params.dataPath if isinstance(params.dataPath, list) else [params.dataPath]
    trainRatio = params.ratioOfTraining
    validationRatio = params.ratioOfValidation
    testRatio = params.ratioOfTesting
    vc = params.versionCreateStructuredTabularClassification
    #stringtoNumeric = params.stringtoNumeric
    #missingValue = params.missingValue
    #outlier = params.outlier    
    stringtoNumeric = {item.columnName: item.numericalTransformation for item in vc.numericalTransformations}
    missingValue = {item.columnName: item.missingValueHandling for item in vc.missingValueHandlings}
    outlier = {item.columnName: item.outlierHandling for item in vc.outlierHandlings}
    status = params.status
    statusMessage = params.statusMessage
    preprocessingResult = params.preprocessingResult

    key = f"preprocessing:{params.projectId}:{params.versionId}"
    try:
        current_status_data = {
            'status': status or 'READY',
            'statusMessage': statusMessage or '요청 값이 모두 들어왔습니다.',
            'ratioOfTraining': params.ratioOfTraining,
            'ratioOfValidation': params.ratioOfValidation,
            'ratioOfTesting': params.ratioOfTesting,
            'versionCreateStructuredTabularClassification': params.versionCreateStructuredTabularClassification.dict()
        }
        r.set(key, json.dumps(current_status_data))        
        exit_code = os.system(
            f"python3 /mnt/dlabflow/structured/backend/kubeflow/pipelines/preprocessing.py "
            f"--projectId={projectId} "
            f"--versionId={versionId} "
            f"--dataPath='{json.dumps(file_paths)}' "
            f"--trainRatio={trainRatio} "
            f"--validationRatio={validationRatio} "
            f"--testRatio={testRatio} "
            f"--stringtoNumeric='{json.dumps(stringtoNumeric)}' "
            f"--missingValue='{json.dumps(missingValue)}' "
            f"--outlier='{json.dumps(outlier)}' "
            f"--status={params.status or ''} "
            f"--statusMessage={params.statusMessage or ''} "
            f"--preprocessingResult={json.dumps(params.preprocessingResult or {})}"
        )        
        if exit_code == 0:
            status = 'RUNNING'
            statusMessage = '데이터 전처리 중입니다.'
        else:
            status = 'ERROR'
            statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
            logger.error(f"{statusMessage}")
            raise
    except Exception:
        status = 'ERROR'
        statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
        logger.error(f"{statusMessage}")
        raise
    current_status_data.update({'status': status, 'statusMessage': statusMessage})
    r.set(key, json.dumps(current_status_data))
    return {
        'projectId': projectId,
        'versionId': versionId,
        'status': status,
        'statusMessage': statusMessage
    }

@svc.api(input=JSON(pydantic_model=PreprocessingParams), output=JSON(), route='/preprocessing/get')
def preprocessing_get(params: PreprocessingParams):
    key = f"preprocessing:{params.projectId}:{params.versionId}"
    error_message = 'BentoML에서 데이터 전처리 중 오류가 발생하였습니다.'

    current_status_json = r.get(key)
    if current_status_json is None:
        current_status_data = {'status': None, 'statusMessage': '들어온 값이 없습니다.'}
    else:
        current_status_data = json.loads(current_status_json)

    if params.status:
        current_status_data['status'] = params.status
        if params.statusMessage:
            current_status_data['statusMessage'] = params.statusMessage
        else:
            if params.status == 'READY':
                current_status_data['statusMessage'] = '요청 값이 모두 들어왔습니다.'
            elif params.status == 'RUNNING':
                current_status_data['statusMessage'] = '데이터 전처리 중입니다.'
            elif params.status == 'FINISH':
                current_status_data['statusMessage'] = '데이터 전처리가 완료되었습니다. '
            elif params.status == 'ERROR':
                current_status_data['statusMessage'] = error_message
    r.set(key, json.dumps(current_status_data))

    if hasattr(params, 'progress') and params.progress is not None:
        current_status_data['progress'] = params.progress

    if hasattr(params, 'trainData') and params.trainData is not None:
        current_status_data['trainData'] = params.trainData

    if hasattr(params, 'validationData') and params.validationData is not None:
        current_status_data['validationData'] = params.validationData

    if hasattr(params, 'testData') and params.testData is not None:
        current_status_data['testData'] = params.testData

    if hasattr(params, 'preprocessingResult') and params.preprocessingResult is not None:
        current_status_data['preprocessingResult'] = params.preprocessingResult

    r.set(key, json.dumps(current_status_data))

    response_json = {
        'projectId': params.projectId,
        'versionId': params.versionId,
        'statusOfPreprocessing': current_status_data.get('status'),
        'numOfTrain': current_status_data.get('trainData'),
        'numOfValidation': current_status_data.get('validationData'),
        'numOfTest': current_status_data.get('testData'),
        'message': current_status_data.get('statusMessage'),
        'structuredTabularClassificationPreProcessingResponse': current_status_data.get('preprocessingResult')
    }

    logger.info(f"{json.dumps(response_json, ensure_ascii=False, indent=4)}")
    return response_json

@svc.api(input=JSON(pydantic_model=TrainingParams), output=JSON(), route='/training/post')
async def training_post(params: TrainingParams):
    projectId = params.projectId
    versionId = params.versionId
    target = params.structuredTabularClassificationTrainingRequestBody.dependentVariable.columnName
    feature = [v.columnName for v in params.structuredTabularClassificationTrainingRequestBody.independentVariables]
    algorithm = params.algorithm
    tuning = params.tuning
    advancedSettingForClassification = params.advancedSettingForClassification
    status = params.status
    statusMessage = params.statusMessage
    trainingResult = params.trainingResult

    key = f"training:{params.projectId}:{params.versionId}"
    try:
        current_status_data = {
            'status': status or 'READY',
            'statusMessage': statusMessage or '요청 값이 모두 들어왔습니다.',
            'algorithm': algorithm,
            'target': target,
            'feature': feature
        }
        r.set(key, json.dumps(current_status_data))
        exit_code = os.system(
            f"python3 /mnt/dlabflow/structured/backend/kubeflow/pipelines/training.py "
            f"--projectId={projectId} "
            f"--versionId={versionId} "
            f"--target={target} "
            f"--feature='{json.dumps(feature)}' "
            f"--algorithm={algorithm} "
            f"--tuning={tuning} "
            f"--advancedSettingForClassification='{json.dumps(advancedSettingForClassification)}' "
            f"--status={status} "
            f"--statusMessage={statusMessage} "
            f"--trainingResult={trainingResult}"
        )
        if exit_code == 0:
            status = 'RUNNING'
            statusMessage = '모델 학습 중입니다.'
        else:
            status = 'ERROR'
            statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
            logger.error(f"{statusMessage}")
            raise
    except Exception:
        status = 'ERROR'
        statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
        logger.error(f"{statusMessage}")
        raise
    current_status_data.update({'status': status, 'statusMessage': statusMessage})
    r.set(key, json.dumps(current_status_data))
    return {
        'projectId': projectId,
        'versionId': versionId,
        'statusOfTrain': status,
        'statusMessage': statusMessage
    }

@svc.api(input=JSON(pydantic_model=TrainingParams), output=JSON(), route='/training/get')
def training_get(params: TrainingParams):
    key = f"training:{params.projectId}:{params.versionId}"
    error_message = 'BentoML에서 모델 학습 중 오류가 발생하였습니다.'

    current_status_json = r.get(key)
    if current_status_json is None:
        current_status_data = {'status': None, 'statusMessage': '들어온 값이 없습니다.'}
    else:
        current_status_data = json.loads(current_status_json)

    if params.status:
        current_status_data['status'] = params.status
        current_status_data['statusMessage'] = params.statusMessage or {
            'READY': '요청 값이 모두 들어왔습니다.',
            'RUNNING': '모델 학습 중입니다.',
            'FINISH': '모델 학습이 완료되었습니다.',
            'ERROR': error_message
        }.get(params.status, '')

    if hasattr(params, 'progress') and params.progress is not None:
        current_status_data['progress'] = params.progress

    if hasattr(params, 'accuracy') and params.accuracy is not None:
        current_status_data['accuracy'] = params.accuracy

    if hasattr(params, 'precision') and params.precision is not None:
        current_status_data['precision'] = params.precision

    if hasattr(params, 'recall') and params.recall is not None:
        current_status_data['recall'] = params.recall

    if hasattr(params, 'auc') and params.auc is not None:
        current_status_data['auc'] = params.auc

    if hasattr(params, 'trainingResult') and params.trainingResult is not None:
        current_status_data['trainingResult'] = params.trainingResult

    r.set(key, json.dumps(current_status_data))

    """
    training_result = current_status_data.get('trainingResult')
    if training_result and isinstance(training_result, dict):
        training_result_list = []
        for key_name, value in training_result.items():
            training_result_list.append({
                key_name: value,
                'columnType': 'STRING'
            })
        current_status_data['trainingResult'] = training_result_list
    """

    response_json = {
        'projectId': params.projectId,
        'versionId': params.versionId,
        'algorithm': params.algorithm,
        'statusOfTrain': current_status_data.get('status'),
        'statusMessage': current_status_data.get('statusMessage'),
        #'progress': current_status_data.get('progress', 0),
        'accuracy': current_status_data.get('accuracy', 0),
        'precisions': current_status_data.get('precision', 0),
        'recall': current_status_data.get('recall', 0),
        'auc': current_status_data.get('auc', 0),        
        'structuredTabularClassificationTrainingResponseBody': current_status_data.get('trainingResult')
    }

    logger.info(f"{json.dumps(response_json, ensure_ascii=False, indent=4)}")
    return response_json

@svc.api(input=JSON(pydantic_model=InferenceParams), output=JSON(), route='/inference/post')
async def inference_post(params: InferenceParams):
    projectId = params.projectId
    versionId = params.versionId
    sessionId = params.sessionId
    inferenceTabularFile = params.inferenceTabularFile
    status = params.status
    statusMessage = params.statusMessage
    inferenceResult = params.inferenceResult

    key = f"inference:{projectId}:{versionId}:{sessionId}"
    try:
        current_status_data = {'status': 'READY', 'statusMessage': '요청 값이 모두 들어왔습니다.'}
        r.set(key, json.dumps(current_status_data))
        exit_code = os.system(
            f"python3 /mnt/dlabflow/structured/backend/kubeflow/pipelines/inference.py "
            f"--projectId={projectId} "
            f"--versionId={versionId} "
            f"--sessionId={sessionId} "
            f"--inferenceTabularFile={inferenceTabularFile} "
            f"--status={status} "
            f"--statusMessage={statusMessage} "
            f"--inferenceResult={inferenceResult}"
        )
        if exit_code == 0:
            status = 'RUNNING'
            statusMessage = '모델 추론 중입니다.'
        else:
            status = 'ERROR'
            statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
            logger.error(f"{statusMessage}")
            raise
    except Exception:
        status = 'ERROR'
        statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
        logger.error(f"{statusMessage}")
        raise
    current_status_data.update({'status': status, 'statusMessage': statusMessage})
    r.set(key, json.dumps(current_status_data))
    return {
        'projectId': projectId,
        'versionId': versionId,
        'sessionId': sessionId,
        'statusOfInference': status,
        'statusMessage': statusMessage
    }

@svc.api(input=JSON(pydantic_model=InferenceParams), output=JSON(), route='/inference/get')
def inference_get(params: InferenceParams):
    key = f"inference:{params.projectId}:{params.versionId}:{params.sessionId}"
    error_message = 'BentoML에서 모델 추론 중 오류가 발생하였습니다.'

    current_status_json = r.get(key)
    if current_status_json is None:
        current_status_data = {'status': None, 'statusMessage': '들어온 값이 없습니다.'}
    else:
        current_status_data = json.loads(current_status_json)

    if params.status:
        current_status_data['status'] = params.status
        current_status_data['statusMessage'] = params.statusMessage or {
            'READY': '요청 값이 모두 들어왔습니다.',
            'RUNNING': '모델 추론 중입니다.',
            'FINISH': '모델 추론이 완료되었습니다.',
            'ERROR': error_message
        }.get(params.status, '')

    if hasattr(params, 'progress') and params.progress is not None:
        current_status_data['progress'] = params.progress

    if hasattr(params, 'inferenceResult') and params.inferenceResult is not None:
        current_status_data['inferenceResult'] = params.inferenceResult

    r.set(key, json.dumps(current_status_data))

    response_json = {
        'projectId': params.projectId,
        'versionId': params.versionId,
        'sessionId': params.sessionId,
        'statusOfInference': current_status_data.get('status'),
        'statusMessage': current_status_data.get('statusMessage'),
        #'progress': current_status_data.get('progress', 0),
        'structuredTabularClassificationInferenceResult': current_status_data.get('inferenceResult')
    }

    logger.info(f"{json.dumps(response_json, ensure_ascii=False, indent=4)}")
    return response_json

@svc.api(input=JSON(pydantic_model=AutoannotationParams), output=JSON(), route='/autoannotation/post')
async def autoannotation_post(params: AutoannotationParams):
    id = params.id
    projectId = params.projectId
    versionId = params.versionId
    datasetId = params.datasetId
    autoAlgorithm = params.autoAlgorithm
    minConfidence = params.minConfidence
    maxConfidence = params.maxConfidence
    target_image_paths = params.targetImagePaths if isinstance(params.targetImagePaths, list) else [params.targetImagePaths]
    classDefinitions = params.classDefinitions
    status = params.status
    statusMessage = params.statusMessage
    autoAnnotationResult = params.autoAnnotationResult

    #key = f"autoannotation:{projectId}:{versionId}"
    key = f"autoannotation:{id}"
    current_status_data = {'status': 'READY', 'statusMessage': '요청 값이 모두 들어왔습니다.'}
    r.set(key, json.dumps(current_status_data))
    exit_code = os.system(
        f"python3 /mnt/dlabflow/unstructured/object_detection/autoannotation.py "
        f"--id={id} "
        f"--projectId={projectId} "
        f"--versionId={versionId} "
        f"--datasetId={datasetId} "
        f"--autoAlgorithm={autoAlgorithm} "
        f"--minConfidence={minConfidence} "
        f"--maxConfidence={maxConfidence} "
        f"--targetImagePaths='{json.dumps(target_image_paths)}' "
        f"--classDefinitions='{json.dumps([c.dict() for c in classDefinitions])}' "
        f"--status={status} "
        f"--statusMessage={statusMessage} "
        f"--autoAnnotationResult={autoAnnotationResult}"
    )
    if exit_code == 0:
        status = 'READY'
        statusMessage = '요청 값이 모두 들어왔습니다.'
    else:
        status = 'ERROR'
        statusMessage = 'Kubeflow에서 컨테이너 생성에 실패하였습니다.'
        current_status_data.update({'status': status, 'statusMessage': statusMessage})
        r.set(key, json.dumps(current_status_data))
        logger.error(f"{statusMessage}")
        raise

    current_status_data.update({'status': status, 'statusMessage': statusMessage})
    r.set(key, json.dumps(current_status_data))

    request_json = {
        'id': id,
        'statusOfAutoAnnotation': status,
        'statusMessage': statusMessage
    }

    logger.info(f"{json.dumps(request_json, ensure_ascii=False, indent=2)}")
    return request_json

@svc.api(input=JSON(pydantic_model=AutoannotationParams), output=JSON(), route='/autoannotation/get')
def autoannotation_get(params: AutoannotationParams):
    #key = f"autoannotation:{params.projectId}:{params.versionId}"
    key = f"autoannotation:{params.id}"
    error_message = '오토어노테이션 진행 중 오류가 발생하였습니다.'

    current_status_json = r.get(key)
    if current_status_json is None:
        current_status_data = {'status': None, 'statusMessage': '들어온 값이 없습니다.'}
    else:
        current_status_data = json.loads(current_status_json)

    if params.status:
        current_status_data['status'] = params.status
        current_status_data['statusMessage'] = params.statusMessage or {
            'READY': '요청 값이 모두 들어왔습니다.',
            'RUNNING': '오토어노테이션 진행 중입니다.',
            'FINISH': '오토어노테이션이 완료되었습니다.',
            'ERROR': error_message
        }.get(params.status, '')

    if hasattr(params, 'progress') and params.progress is not None:
        current_status_data['progress'] = params.progress

    if hasattr(params, 'autoAnnotationResult') and params.autoAnnotationResult is not None:
        current_status_data['autoAnnotationResult'] = params.autoAnnotationResult

    r.set(key, json.dumps(current_status_data))

    response_json = {
        'id': params.id,
        'statusOfAutoAnnotation': current_status_data.get('status'),
        'statusMessage': current_status_data.get('statusMessage'),
        #'progress': current_status_data.get('progress', 0),
        'resultLabelingPaths': current_status_data.get('autoAnnotationResult')
    }

    logger.info(f"{json.dumps(response_json, ensure_ascii=False, indent=2)}")
    return response_json

