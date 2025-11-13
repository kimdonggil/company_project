import os
from functools import partial
from kfp.components import create_component_from_func
from kfp import onprem
import argparse
import kfp
import requests
from dotenv import load_dotenv, dotenv_values

dotenv_path = '/mnt/dlabflow/structured/config'
load_dotenv(dotenv_path)

KubeflowPieplineEDA = os.getenv('KubeflowPieplineEDA')
KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')
KubeflowVolumeName = os.getenv('KubeflowVolumeName1')
KubeflowVolumeMountPath = os.getenv('KubeflowVolumeMountPath1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:structured-20250924-latest')
def EDA(realFileId: int, filePath: str, status: str, statusMessage: str, edaResult: str):
    import os
    import pandas as pd
    from tqdm import tqdm
    import time
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
    MinioBucketEDA = os.getenv('MinioBucketEDA')
    BentomlEDAGet = os.getenv('BentomlEDAGet')

    base_paths = f"/{MinIODefaultPath}/{MinioBucketUser}/{realFileId}/{MinioBucketEDA}"
    os.makedirs(base_paths, exist_ok=True)

    client = Minio(
        endpoint=MinioEndpoint,        
        access_key=MinioAccessKey,        
        secret_key=MinioSecretKey,        
        secure=MinioSecure    
    )

    def send_status_to_bentoml(realFileId, filePath, status, statusMessage=None, edaResult=None):
        try:
            url = BentomlEDAGet
            payload = {
                'realFileId': realFileId,
                'status': status,
            }
            if filePath:
                payload['filePath'] = filePath
            if statusMessage:
                payload['statusMessage'] = statusMessage
            if edaResult:
                payload['edaResult'] = edaResult
            requests.post(url, json=payload, timeout=3)
        except Exception:
            statusMessage = 'EDA 결과를 전송하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(realFileId, filePath, status='ERROR', statusMessage=statusMessage)
            raise

    def eda_create(data_path, chunksize):
        try:
            parts = data_path.split('/', 1)
            bucket_name = parts[0]
            object_name = parts[1]

            if not filePath:
                statusMessage = 'FilePath 값이 비어 있습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(realFileId, filePath, status='ERROR', statusMessage=statusMessage)
                raise

            parts = filePath.split('/', 1)
            bucket_name = parts[0]
            object_name = parts[1]
            try:
                client.stat_object(bucket_name, object_name)
            except Exception:
                statusMessage = 'MinIO에EDA 파일이 존재하지 않습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(realFileId, filePath, status='ERROR', statusMessage=statusMessage)
                raise

            response = client.get_object(bucket_name, object_name)
            try:
                reader = pd.read_csv(response, chunksize=chunksize, encoding='cp949')
            except Exception:
                statusMessage = 'CSV 파일의 한글 인코딩을 읽을 수 없습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(realFileId, filePath, status='ERROR', statusMessage=statusMessage)
                raise            
            chunks = []
            chunk_count = 0
            for i, chunk in enumerate(reader, 1):
                chunks.append(chunk)
                chunk_count += 1

            df_row = pd.concat(chunks, ignore_index=True)

            drop_cols = []
            for col in df_row.select_dtypes(include='object').columns:
                prefixes = df_row[col].str.extract(r'([A-Za-z]+)')[0]
                unique_count = prefixes.nunique()
                if unique_count >= 10:
                    drop_cols.append(col)
            logger.info(f"{drop_cols}")
            df = df_row.drop(columns=drop_cols)

            if df.empty or df.shape[1] == 0:
                statusMessage = 'CSV 파일에 유효한 데이터가 없습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(realFileId, filePath, status='ERROR', statusMessage=statusMessage)
                raise

            num_vars = df.shape[1]
            num_obs = df.shape[0]
            duplicate_rows = df.duplicated().sum()
            duplicate_percent = round(duplicate_rows / num_obs * 100, 1)
            def sizeof_fmt(num, suffix='B'):
                for unit in ['','K','M','G','T','P','E','Z']:
                    if abs(num) < 1024.0:
                        return f"{num:3.1f}{unit}{suffix}"
                    num /= 1024.0
                return f"{num:.1f}Y{suffix}"
            stat = client.stat_object(bucket_name, object_name)
            file_size_bytes = stat.size
            file_size_human = sizeof_fmt(file_size_bytes)
            file_size_kb = round(file_size_bytes / 1024, 1)
            dataset_stats = pd.DataFrame({
                'dataStatistics': [
                    'numberOfColumns',
                    'numberOfRows',
                    'numberOfDuplicateRows',
                    'duplicateRowRatio',
                    'dataSize',
                ],
                'dataStatisticsValues': [
                    num_vars,
                    num_obs,
                    duplicate_rows,
                    f"{duplicate_percent}",
                    file_size_kb
                ]
            })
            num_numeric = df.select_dtypes(include=['number']).shape[1]
            num_text = df.select_dtypes(include=['object']).shape[1]
            variable_types = pd.DataFrame({
                'columnTypes': ['NUMERIC', 'STRING'],
                'columnTypesValues': [num_numeric, num_text]
            })
            variable_types['type'] = variable_types['columnTypesValues'].apply(
                lambda x: 'NUMERIC' if isinstance(x, (int, float)) else 'STRING'
            )
            missing_per_column = df.isnull().sum()
            missing_percent_per_column = (missing_per_column / num_obs * 100).round(1)
            missing_stats = pd.DataFrame({
                'columnName': df.columns,
                'missingValue': missing_per_column.values,
            })
            missing_stats['type'] = [ 'NUMERIC' if pd.api.types.is_numeric_dtype(df[col]) else 'STRING' for col in df.columns ]
            dataset_stats.to_csv(f"/{base_paths}/dataset_stats.csv", index=False, encoding='utf-8-sig')
            variable_types.to_csv(f"/{base_paths}/variable_types.csv", index=False, encoding='utf-8-sig')
            missing_stats.to_csv(f"/{base_paths}/missing_stats.csv", index=False, encoding='utf-8-sig')
        except Exception:
            statusMessage = 'Kubeflow에서 EDA 처리 중 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(realFileId, filePath, status='ERROR', statusMessage=statusMessage)
            raise

    """ Task """
    logger.info(f"realFileId: {realFileId}")
    send_status_to_bentoml(realFileId, filePath, status='READY', statusMessage='요청 값이 모두 들어왔습니다.')
    send_status_to_bentoml(realFileId, filePath, status='RUNNING', statusMessage='EDA 처리 중입니다.')
    eda_create(data_path=filePath, chunksize=10000)
    send_status_to_bentoml(realFileId, filePath, status='FINISH', statusMessage='EDA 처리가 완료되었습니다.')

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--realFileId', type=int)
    parser.add_argument('--filePath', type=str)
    parser.add_argument('--status', type=str)
    parser.add_argument('--statusMessage', type=str)
    parser.add_argument('--edaResult', type=str)
    args = parser.parse_args()
    EDA_task = EDA(args.realFileId, args.filePath, args.status, args.statusMessage, args.edaResult) \
        .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
        .execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = f"{KubeflowPieplineEDA}_pipelines.zip"
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
    experiment = client.create_experiment(name=f"{KubeflowPieplineEDA}")
    run = client.run_pipeline(experiment.id, f"{KubeflowPieplineEDA} pipelines", pipeline_package_path)
