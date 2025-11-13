import os
from functools import partial
from kfp.components import create_component_from_func
from kfp import onprem
import argparse
import kfp
import requests
import json
from typing import List
from dotenv import load_dotenv, dotenv_values

dotenv_path = '/mnt/dlabflow/structured/config'
load_dotenv(dotenv_path)

KubeflowPieplineInference = os.getenv('KubeflowPieplineInference')
KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')
KubeflowVolumeName = os.getenv('KubeflowVolumeName1')
KubeflowVolumeMountPath = os.getenv('KubeflowVolumeMountPath1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:structured-20251016-latest')
def Inference(projectId: str, versionId: str, sessionId: str, inferenceTabularFile: str, status: str, statusMessage: str, inferenceResult: str):
    import os
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from minio import Minio
    import requests
    from dotenv import load_dotenv, dotenv_values
    import json
    import traceback
    import logging
    import redis
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, make_scorer
    import plotly.graph_objects as go
    import lightgbm as lgb
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    import base64
    from io import StringIO, BytesIO
    import joblib
    
    logger = logging.getLogger('Kubeflow Log')
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
    r = redis.Redis(host=RedisHost, port=RedisPort, db=RedisDB)

    MinIODefaultPath = os.getenv('MinIODefaultPath')
    MinioEndpoint = os.getenv('MinioEndpoint')
    MinioAccessKey = os.getenv('MinioAccessKey')
    MinioSecretKey = os.getenv('MinioSecretKey')
    MinioSecure = os.getenv('MinioSecure', 'False').lower() == 'true'
    MinioBucketUser = os.getenv('MinioBucketUser1')
    MinioBucketPreprocessing = os.getenv('MinioBucketPreprocessing')
    MinIOBucketTraining = os.getenv('MinIOBucketTraining')
    MinIOBucketInference = os.getenv('MinIOBucketInference')
    BentomlInferenceGet = os.getenv('BentomlInferenceGet')

    preprocessing_key = f"preprocessing:{projectId}:{versionId}"
    preprocessing_data_json = r.get(preprocessing_key)
    if preprocessing_data_json:
        preprocessing_data = json.loads(preprocessing_data_json)
        stringto_numeric_ = preprocessing_data.get('versionCreateStructuredTabularClassification', {}).get('numericalTransformations', [])
        missing_value_ = preprocessing_data.get('versionCreateStructuredTabularClassification', {}).get('missingValueHandlings', [])
        outlier_ = preprocessing_data.get('versionCreateStructuredTabularClassification', {}).get('outlierHandlings', [])
    training_key = f"training:{projectId}:{versionId}"
    training_data_json = r.get(training_key)
    if training_data_json:
        training_data = json.loads(training_data_json)
        target = training_data.get('target', [])
        feature = training_data.get('feature', [])
        algorithm = training_data.get('algorithm', [])
        
    stringto_numeric_config = {item['columnName']: item['numericalTransformation'] for item in stringto_numeric_}
    missing_value_config = {item['columnName']: item['missingValueHandling'] for item in missing_value_}
    outlier_config = {item['columnName']: item['outlierHandling'] for item in outlier_}

    base_paths = f"/{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinIOBucketInference}/"
    training_paths = f"/{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinIOBucketTraining}/"
    model_file_path = os.path.join(training_paths, 'model_weight', f"{algorithm}_model.pkl")
    os.makedirs(base_paths, exist_ok=True)

    client = Minio(
        endpoint=MinioEndpoint,
        access_key=MinioAccessKey,
        secret_key=MinioSecretKey,
        secure=MinioSecure
    )

    def send_status_to_bentoml(projectId, versionId, sessionId, status, statusMessage=None, inferenceResult=None, progress=None):
        try:
            url = BentomlInferenceGet
            payload = {
                'projectId': projectId,
                'versionId': versionId,
                'sessionId': sessionId,
                'status': status,
            }
            if statusMessage:
                payload['statusMessage'] = statusMessage
            if inferenceResult is not None:
                payload['inferenceResult'] = inferenceResult
            if progress is not None:
                payload['progress'] = progress            
            requests.post(url, json=payload, timeout=3)
        except Exception:
            statusMessage = '추론 결과를 전송하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
            raise        

    def csv_reader(chunksize):
        try:
            decoded_bytes = base64.b64decode(inferenceTabularFile)
            decoded_str = decoded_bytes.decode('utf-8')
            reader = pd.read_csv(StringIO(decoded_str), chunksize=chunksize)
            chunks = []
            for i, chunk in enumerate(reader, 1):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
            return data        
        except Exception:
            statusMessage = '추론 데이터를 불러오는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
            raise

    def string_to_numeric_with_mapping(data, config):
        le_dict = {}
        le_mapping_dict = {}
        try:
            for col, method in config.items():
                if not method:
                    continue
                if method.lower() == 'one_hot_encoding':
                    dummy_na = data[col].isna().any()
                    dummies = pd.get_dummies(data[col], prefix=col, dummy_na=dummy_na)
                    dummies = dummies.astype(int)
                    le_mapping_dict.update({c: {val: 1 if val in c else 0 for val in data[col].dropna().unique()} for c in dummies.columns})
                    data = pd.concat([data.drop(col, axis=1), dummies], axis=1)
                elif method.lower() == 'label_encoding':
                    le = LabelEncoder()
                    not_null = data[col][data[col].notna()]
                    le.fit(not_null)
                    encoded = data[col].copy()
                    encoded[data[col].notna()] = le.transform(not_null)
                    encoded = encoded.astype(float)
                    data[col] = encoded
                    le_dict[col] = le
                    le_mapping_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        except Exception:
            statusMessage = '범주형 데이터를 수치화하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
            raise
        return data, le_dict, le_mapping_dict

    def missing_value(data, config, show_plot=False):
        if show_plot:
            missing_cols = [col for col in data.columns if data[col].isnull().sum() > 0]
            if missing_cols:
                fig = go.Figure()
                for col in missing_cols:
                    fig.add_trace(
                        go.Bar(
                            x=[col],
                            y=[data[col].isnull().sum()],
                            name=col,
                            showlegend=False
                        )
                    )
                fig.update_layout(
                    title='Missing Values',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.show()
        le = LabelEncoder()
        for col, method in config.items():
            if not method:
                continue
            method = method.lower()
            try:
                if method.lower() == 'mean_imputation':
                    value = data[col].mean()
                    data[col].fillna(value, inplace=True)
                elif method.lower() == 'median_imputation':
                    value = data[col].median()
                    data[col].fillna(value, inplace=True)
                elif method.lower() == 'mode_imputation':
                    value = data[col].mode().iloc[0]
                    data[col].fillna(value, inplace=True)
                elif method.lower() == 'delete':
                    data = data.dropna(subset=[col], axis=0)                    
            except Exception:
                statusMessage = '결측값을 대체하는 과정에서 오류가 발생하였습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
                raise
        return data

    def outliers(data, config, selected_cols=None, show_plot=False):
        if show_plot:
            if selected_cols is None:
                selected_cols = list(config.keys())
            def get_outliers(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                return series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
            cols_with_outliers = [col for col in selected_cols if col in data.columns and len(get_outliers(data[col])) > 0]
            if cols_with_outliers:
                fig = go.Figure()
                for col in cols_with_outliers:
                    fig.add_trace(go.Box(x=data[col], name=col, orientation='h', boxpoints='outliers'))
                fig.update_layout(
                    title='Outlier',
                    height=300 + 50*len(cols_with_outliers),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                fig.show()
        outlier_mapping_dict = {}
        for col, method in config.items():
            if not method:
                continue
            method = method.lower()
            try:
                if method.lower() == 'iqr_removal':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    before_rows = data.shape[0]
                    data = data[(data[col] >= Q1-1.5*IQR) & (data[col] <= Q3+1.5*IQR)]
                    after_rows = data.shape[0]
                    outlier_mapping_dict[col] = before_rows - after_rows
            except Exception:
                statusMessage = '이상치를 삭제하는 과정에서 오류가 발생하였습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
                raise
        return data, outlier_mapping_dict

    def pie_chart(df, target_col, save_path):
        counts = df[target_col].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.3))
        fig.update_layout(
            title=f'Submission {target_col} Distribution',
            width=700,
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        with open(save_path, 'rb') as f:
            chart_base64 = base64.b64encode(f.read()).decode('utf-8')
        return chart_base64

    def encode_file_to_base64(file_path):
        try:
            with open(file_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            statusMessage = '데이터의 Base64 인코딩 과정에서 오류가 발생했습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
            raise

    """ Task """
    logger.info(f"projectId: {projectId}")
    logger.info(f"versionId: {versionId}")
    logger.info(f"sessionId: {sessionId}")
    #logger.info(f"string to numeric: {stringto_numeric_config}")
    #logger.info(f"missing value: {missing_value_config}")
    #logger.info(f"outlier: {outlier_config}")    
    send_status_to_bentoml(projectId, versionId, sessionId, status='READY', statusMessage='요청 값이 모두 들어왔습니다.', progress=0)
    send_status_to_bentoml(projectId, versionId, sessionId, status='RUNNING', statusMessage='모델 추론 중입니다.')
    df_row = csv_reader(chunksize=10000)

    """
    categorical_columns = df_row[feature].select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        statusMessage = f"추론 대상 데이터에서 {categorical_columns}에 범주형 데이터가 존재하여, 추론 모델에 사용할 수 없습니다. 버전 생성 단계에서 해당 변수를 수치화하는 작업이 필요합니다."
        logger.error(f"{statusMessage}")
        send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
        raise

    missing_columns = df_row[feature].columns[df_row[feature].isnull().any()].tolist()
    if missing_columns:
        statusMessage = f"추론 대상 데이터에서 {missing_columns}에 결측값이 존재하여, 추론 모델에 사용할 수 없습니다. 버전 생성 단계에서 해당 변수를 전처리하는 작업이 필요합니다."
        logger.error(f"{statusMessage}")
        send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
        raise
    """

    if all(f in df_row.columns for f in feature):
        if target not in feature:
            df_clean = df_row[feature]
            send_status_to_bentoml(projectId, versionId, sessionId, status='RUNNING', statusMessage='모델 추론 중입니다.', progress=20)
            df_pre_1, le_dict, le_mapping_dict = string_to_numeric_with_mapping(data=df_clean, config=stringto_numeric_config)
            send_status_to_bentoml(projectId, versionId, sessionId, status='RUNNING', statusMessage='모델 추론 중입니다.', progress=40)
            df_pre_2 = missing_value(data=df_pre_1, config=missing_value_config)
            send_status_to_bentoml(projectId, versionId, sessionId, status='RUNNING', statusMessage='모델 추론 중입니다.', progress=60)
            df_pre_3, outlier_mapping_dict = outliers(data=df_pre_2, config=outlier_config)
            df_pre_3.dropna(inplace=True)
            df_final = df_pre_3.copy()

            if not os.path.exists(model_file_path):
                statusMessage = '모델 가중치가 존재하지 않습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
                raise

            model = joblib.load(model_file_path)
            y_pred = model.predict(df_final.values)
            submission = df_final.copy()
            submission.loc[:, target] = y_pred
            submission.to_csv(f"{base_paths}/inference.csv", index=False)

            """ label encoding → decoding """
            """
            submission_original = submission.copy()
            for col, le in le_dict.items():
                if col in submission_original.columns:
                    submission_original[col] = le.inverse_transform(submission_original[col].astype(int))
            submission_original.to_csv(f"{base_paths}/inference_original.csv", index=False)
            """

            submission_csv_path = os.path.join(base_paths, 'inference.csv')
            object_name = f"{projectId}/{versionId}/{MinIOBucketInference}/inference.csv"
            inference_result = {
                'base64InferenceCSV': {
                    'name': 'inference_result.csv',
                    'base64': encode_file_to_base64(f"{base_paths}/inference.csv")
                }
            }
            send_status_to_bentoml(projectId, versionId, sessionId, status='FINISH', inferenceResult=inference_result, statusMessage='모델 추론이 완료되었습니다.')
        else:
            statusMessage = f"학습 시 사용된 종속 변수 '{target}' 이(가) 추론 데이터에 포함되어 있어 모델 추론을 진행할 수 없습니다."
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
            raise
    else:
        statusMessage = f"학습 시 사용된 독립 변수 '{feature}' 이(가) 추론 데이터에 포함되지 않아 모델 추론을 진행할 수 없습니다."
        logger.error(f"{statusMessage}")
        send_status_to_bentoml(projectId, versionId, sessionId, status='ERROR', statusMessage=statusMessage)
        raise        

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--sessionId', type=str)
    parser.add_argument('--inferenceTabularFile', type=str)
    parser.add_argument('--status', type=str)
    parser.add_argument('--statusMessage', type=str)
    parser.add_argument('--inferenceResult', type=str)
    args = parser.parse_args()
    Inference_task = Inference(args.projectId, args.versionId, args.sessionId, args.inferenceTabularFile, args.status, args.statusMessage, args.inferenceResult) \
        .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
        .execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = f"{KubeflowPieplineInference}_pipelines.zip"
    kfp.compiler.Compiler().compile(pipelines, pipeline_package_path)
    HOST = f"http://{KubeflowHost}"
    USERNAME = KubeflowUsername
    PASSWORD = KubeflowPassword
    NAMESPACE = KubeflowNamespace
    session = requests.Session()
    response = session.get(HOST)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'login': USERNAME, 'password': PASSWORD}
    session.post(response.url, headers = headers, data = data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    client = kfp.Client(host = f"{HOST}/pipeline", cookies = f"authservice_session={session_cookie}", namespace = NAMESPACE)
    experiment = client.create_experiment(name=f"{KubeflowPieplineInference}")
    run = client.run_pipeline(experiment.id, f"{KubeflowPieplineInference} pipelines", pipeline_package_path)
