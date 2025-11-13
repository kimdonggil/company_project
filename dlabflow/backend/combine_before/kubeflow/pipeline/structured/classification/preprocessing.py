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

KubeflowPieplinePreprocessing = os.getenv('KubeflowPieplinePreprocessing')
KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')
KubeflowVolumeName = os.getenv('KubeflowVolumeName1')
KubeflowVolumeMountPath = os.getenv('KubeflowVolumeMountPath1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:structured-20251016-latest')
def Preprocessing(projectId: str, versionId: str, dataPath: List[str], trainRatio: int, validationRatio: int, testRatio: int, stringtoNumeric: str, missingValue: str, outlier: str, status: str, statusMessage: str, preprocessingResult: str):
    import os
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from minio import Minio
    import requests
    from dotenv import load_dotenv, dotenv_values
    import json
    import traceback
    import logging

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

    MinIODefaultPath = os.getenv('MinIODefaultPath')
    MinioEndpoint = os.getenv('MinioEndpoint')
    MinioAccessKey = os.getenv('MinioAccessKey')
    MinioSecretKey = os.getenv('MinioSecretKey')
    MinioSecure = os.getenv('MinioSecure', 'False').lower() == 'true'
    MinioBucketUser = os.getenv('MinioBucketUser1')
    MinioBucketPreprocessing = os.getenv('MinioBucketPreprocessing')
    BentomlPreprocessingGet = os.getenv('BentomlPreprocessingGet')

    base_paths = f"/{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinioBucketPreprocessing}/"
    os.makedirs(base_paths, exist_ok=True)

    client = Minio(
        endpoint=MinioEndpoint,
        access_key=MinioAccessKey,
        secret_key=MinioSecretKey,
        secure=MinioSecure
    )

    def send_status_to_bentoml(projectId, versionId, dataPath, status, statusMessage=None, trainData=None, validationData=None, testData=None, preprocessingResult=None, progress=None):
        try:
            url = BentomlPreprocessingGet
            payload = {
                'projectId': projectId,
                'versionId': versionId,
                'status': status,
            }
            if dataPath:
                payload['dataPath'] = dataPath
            if statusMessage:
                payload['statusMessage'] = statusMessage
            if preprocessingResult is not None:
                payload['preprocessingResult'] = preprocessingResult
            if progress is not None:
                payload['progress'] = progress
            if trainData is not None:
                payload['trainData'] = trainData
            if validationData is not None:
                payload['validationData'] = validationData
            if testData is not None:
                payload['testData'] = testData
            requests.post(url, json=payload, timeout=3)
        except Exception:
            statusMessage = '전처리 결과를 전송하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
            raise

    def csv_reader(data_paths, chunksize):
        all_chunks = []
        try:
            for data_path in data_paths:
                parts = data_path.split('/', 1)
                bucket_name = parts[0]
                object_name = parts[1]
                response = client.get_object(bucket_name, object_name)
                reader = pd.read_csv(response, chunksize=chunksize)
                for chunk in reader:
                    all_chunks.append(chunk)
            data = pd.concat(all_chunks, ignore_index=True)
            logger.info(f"{data.columns}")
            return data
        except Exception as e:
            statusMessage = f'전처리 데이터를 불러오는 중 오류가 발생하였습니다: {str(e)}'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
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
            send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
            raise

        return data, le_dict, le_mapping_dict

    def expand_missing_value_config_after_encoding(data, config):
        expanded_config = {}
        for col, method in config.items():
            if col in data.columns:
                expanded_config[col] = method
            else:
                prefixed_cols = [c for c in data.columns if c.startswith(f"{col}_")]
                for pcol in prefixed_cols:
                    expanded_config[pcol] = method
        return expanded_config

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
                else:
                    statusMessage = f"{col}의 결측치 처리 방법 '{method}'을(를) 지원하지 않습니다."
                    logger.error(f"{statusMessage}")
                    send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
                    raise
            except Exception:
                statusMessage = '결측값을 대체하는 과정에서 오류가 발생하였습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
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
                else:
                    statusMessage = f"{col}의 이상치 처리 방법 '{method}'을(를) 지원하지 않습니다."
                    logger.error(f"{statusMessage}")
                    send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
                    raise
            except Exception:
                statusMessage = '이상치를 삭제하는 과정에서 오류가 발생하였습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
                raise
        return data, outlier_mapping_dict

    """ Task """
    logger.info(f"projectId: {projectId}")
    logger.info(f"versionId: {versionId}")
    send_status_to_bentoml(projectId, versionId, dataPath, status='READY', statusMessage='요청 값이 모두 들어왔습니다.', progress=0)
    send_status_to_bentoml(projectId, versionId, dataPath, status='RUNNING', statusMessage='데이터 전처리 중입니다.')
    df_row = csv_reader(data_paths=dataPath, chunksize=10000)
    drop_cols = []
    for col in df_row.select_dtypes(include='object').columns:
        prefixes = df_row[col].str.extract(r'([A-Za-z]+)')[0]
        unique_count = prefixes.nunique()
        if unique_count >= 10:
            drop_cols.append(col)
    logger.info(f"Delete features: {drop_cols}")
    df_clean = df_row.drop(columns=drop_cols)
    send_status_to_bentoml(projectId, versionId, dataPath, status='RUNNING', statusMessage='데이터 전처리 중입니다.', progress=20)
    stringtoNumeric_dict = json.loads(stringtoNumeric)
    df_pre_1, le_dict, le_mapping_dict = string_to_numeric_with_mapping(data=df_clean, config=stringtoNumeric_dict)
    logger.info(f"string to numeric: {df_pre_1.columns}")
    send_status_to_bentoml(projectId, versionId, dataPath, status='RUNNING', statusMessage='데이터 전처리 중입니다.', progress=40)
    #missingValue_dict = json.loads(missingValue)
    missingValue_dict = expand_missing_value_config_after_encoding(df_pre_1, json.loads(missingValue))
    df_pre_2 = missing_value(data=df_pre_1, config=missingValue_dict)
    logger.info(f"missing value: {df_pre_2.columns}")
    send_status_to_bentoml(projectId, versionId, dataPath, status='RUNNING', statusMessage='데이터 전처리 중입니다.', progress=60)
    outlier_dict = json.loads(outlier)
    df_pre_3, outlier_mapping_dict = outliers(data=df_pre_2, config=outlier_dict)
    logger.info(f"outlier: {df_pre_3.columns}")

    missing_columns = df_pre_3.columns[df_pre_3.isnull().any()].tolist()
    if missing_columns:
        statusMessage = f"전처리 데이터에 결측값이 존재합니다. 모델 훈련을 진행하기 위해서는 {missing_columns} 컬럼의 결측값을 먼저 처리해 주시기 바랍니다."
        logger.error(f"{statusMessage}")
        send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
        raise

    """
    categorical_columns = df_pre_3.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        statusMessage = f"전처리 데이터에 범주형이 존재합니다. 모델 훈련을 진행하기 위해서는 {categorical_columns} 컬럼을 먼저 수치화해 주시기 바랍니다."
        logger.error(f"{statusMessage}")
        send_status_to_bentoml(projectId, versionId, dataPath, status='ERROR', statusMessage=statusMessage)
        raise
    """

    """
    df_final = df_pre_3.copy()
    object_cols = df_final.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        object_cols_dict = {col: 'LABEL_ENCODING' for col in object_cols}
        df_final, le_dict, le_mapping_dict = string_to_numeric_with_mapping(data=df_final, config=object_cols_dict)
    for col in df_final.columns:
        missing_count = df_final[col].isna().sum()
        if missing_count > 0:
            mode_value = df_final[col].mode()[0]
            df_final[col].fillna(mode_value, inplace=True)
    """

    df_pre_3.to_csv(f"{base_paths}/preprocessing.csv")
    total = len(df_pre_3)
    val_count = test_count = round(total * validationRatio/100)
    train_count = total - val_count - test_count
    df_train, df_temp = train_test_split(df_pre_3, train_size=train_count, random_state=42)

    if len(df_temp) % 2 == 0:
        half = len(df_temp) // 2
        df_val = df_temp.iloc[:half]
        df_test = df_temp.iloc[half:]
    else:
        half = len(df_temp) // 2
        df_val = df_temp.iloc[:half]
        df_test = df_temp.iloc[half:half+half]
        df_train = pd.concat([df_train, df_temp.iloc[half+half:]], ignore_index=True)

    train_rows = len(df_train)
    validation_rows = len(df_val)
    test_rows = len(df_test)

    """
    train_rows = round(total_rows * trainRatio/100)
    validation_rows = round(total_rows * validationRatio/100)
    test_rows = total_rows - train_rows - validation_rows
    """

    """
    total_ratio = trainRatio + validationRatio + testRatio
    train_rows = round(total_rows * trainRatio)
    validation_rows = round(total_rows * validationRatio)
    test_rows = total_rows - train_rows - validation_rows
    """

    send_status_to_bentoml(projectId, versionId, dataPath, status='RUNNING', statusMessage='데이터 전처리 중입니다.', progress=80)
    col = df_pre_3.columns.to_list()
    col_final = list({c.split('_')[0] for c in col})
    preprocessed_columns = []
    for col in col_final:
        if col in df_pre_3.columns:
            if pd.api.types.is_object_dtype(df_pre_3[col]):
                col_type = 'STRING'
            else:
                col_type = 'NUMERIC'
        else:
            col_type = 'NUMERIC'
        preprocessed_columns.append(
            {
                'columnName': col,
                'columnType': col_type
            }
        )
    preprocessing_result = {
        'preprocessedColumns': preprocessed_columns
    }
    send_status_to_bentoml(projectId, versionId, dataPath, status='FINISH', statusMessage='데이터 전처리가 완료되었습니다.', preprocessingResult=preprocessing_result, progress=100, trainData=train_rows, validationData=validation_rows, testData=test_rows)

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--dataPath', type=str)
    parser.add_argument('--trainRatio', type=int)
    parser.add_argument('--validationRatio', type=int)
    parser.add_argument('--testRatio', type=int)
    parser.add_argument('--stringtoNumeric', type=str)
    parser.add_argument('--missingValue', type=str)
    parser.add_argument('--outlier', type=str)
    parser.add_argument('--status', type=str)
    parser.add_argument('--statusMessage', type=str)
    parser.add_argument('--preprocessingResult', type=str)
    args = parser.parse_args()
    file_paths = json.loads(args.dataPath) if args.dataPath.startswith('[') else [args.dataPath]
    Preprocessing_task = Preprocessing(args.projectId, args.versionId, file_paths, args.trainRatio, args.validationRatio, args.testRatio, args.stringtoNumeric, args.missingValue, args.outlier, args.status, args.statusMessage, args.preprocessingResult) \
        .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
        .execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = f"{KubeflowPieplinePreprocessing}_pipelines.zip"
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
    experiment = client.create_experiment(name=f"{KubeflowPieplinePreprocessing}")
    run = client.run_pipeline(experiment.id, f"{KubeflowPieplinePreprocessing} pipelines", pipeline_package_path)
