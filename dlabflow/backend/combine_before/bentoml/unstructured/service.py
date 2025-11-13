import os
import io
import ast
import json
from io import BytesIO
from pydantic import BaseModel
from typing import Optional, List
import bentoml
from bentoml.io import JSON, File
from bentoml.exceptions import BentoMLException
import pandas as pd
import xml.etree.ElementTree as ET
import shutil
import aiofiles
import aiohttp
from minio import Minio
from minio.error import S3Error
import pymysql
import subprocess
from contextlib import contextmanager
import logging
import asyncio
import shlex
import dask.dataframe as dd
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor
import math
import warnings
import redis
from tqdm.asyncio import tqdm_asyncio
from datetime import timedelta
warnings.filterwarnings('ignore')

bucket = 'grit'
minio_path = '/mnt/dlabflow/backend/minio'
base_path = os.path.join(minio_path, bucket)

client = Minio('10.40.217.236:9002', 'dlab-backend', 'dlab-backend-secret', secure=False)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n\n', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
bentoml_logger = logging.getLogger("bentoml")
if not bentoml_logger.hasHandlers():
    bentoml_logger.addHandler(ch)
bentoml_logger.setLevel(logging.INFO)

executor = ThreadPoolExecutor(max_workers=10)

svc = bentoml.Service('datasource_preprocessing')

class DataSourceParams(BaseModel):
    projectId: str
    versionId: str    
    folder: str
    filename: str
    path: str
    database: str
    width: int
    height: int
    depth: int
    segmented: int
    name: str
    obj_id: int
    pose: str
    truncated: Optional[int]
    difficult: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    totalPages: int  

class PreprocessingParams(BaseModel):
    projectId: str
    versionId: str
    dataPath: List[str]
    dataNormalization: List[str]
    dataAugmentation: List[str]
    trainRatio: int
    validationRatio: int
    testRatio: int

class TrainingParams(BaseModel):
    projectId: str
    versionId: str
    algorithm: str
    batchsize: int
    epoch: int

class ModelTunningOptions(BaseModel):
    patience: float
    imgSize: int
    optimizer: str
    multiScale: Optional[bool] = False
    cosLr: Optional[bool] = False
    closeMosaic: float
    amp: Optional[bool] = False
    freeze: int
    lr0: float
    lrf: float
    momentum: float
    weightDecay: float
    warmupEpochs: float
    warmupMomentum: Optional[bool] = False
    warmupBiasLr: Optional[bool] = False
    box: float
    cls: float
    dropout: float

class ModelTunningParams(BaseModel):
    projectId: str
    versionId: str
    algorithm: str
    batchsize: int
    epoch: int    
    tuning: Optional[bool] = False
    advancedSettingForObjectDetection: Optional[ModelTunningOptions] = None

class InferenceParams(BaseModel):
    projectId: str
    versionId: str
    sessionId: str

@contextmanager
def mysql_connection():
    db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
    cursor = db.cursor()
    try:
        yield db, cursor
    finally:
        cursor.close()
        db.close()

def db_mysql_dataframe(sql_select):
    with mysql_connection() as (db, cursor):
        df = pd.read_sql(sql_select, db)
    return df

def db_mysql_update(sql_select, updates: dict, conditions: dict):
    with mysql_connection() as (db, cursor):
        set_clause = ', '.join([f"{k}=%s" for k in updates.keys()])
        where_clause = ' AND '.join([f"{k}=%s" for k in conditions.keys()])
        sql = f"UPDATE {sql_select} SET {set_clause} WHERE {where_clause}"
        val = tuple(updates.values()) + tuple(conditions.values())
        try:
            cursor.execute(sql, val)
            db.commit()
        except Exception as e:
            db.rollback()
            bentoml_logger.error(f"{sql_select} 테이블 업데이트 실패: {e}")
            raise BentoMLException(f"{sql_select} 테이블 업데이트에 실패했습니다.")

async def download_from_minio_parallel(bucket: str, prefix: str, target_dir: str):
    def get_presigned_url(object_name):
        return client.presigned_get_object(bucket, object_name, expires=timedelta(seconds=3600))
    async def prepare_download_list():
        loop = asyncio.get_event_loop()
        objects = await loop.run_in_executor(None, lambda: list(client.list_objects(bucket, prefix=prefix, recursive=True)))
        presigned_data = []
        for obj in objects:
            url = await loop.run_in_executor(None, get_presigned_url, obj.object_name)
            local_path = os.path.join(target_dir, os.path.relpath(obj.object_name, prefix))
            presigned_data.append((url, local_path))
        return presigned_data
    async def download_file(session, url, local_path):
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            async with session.get(url) as response:
                if response.status != 200:
                    return local_path, f"HTTP 오류: {response.status}"
                with open(local_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024 * 64)
                        if not chunk:
                            break
                        f.write(chunk)
            return local_path, None
        except Exception as e:
            return local_path, str(e)
    presigned_data = await prepare_download_list()
    bentoml_logger.info(f"MinIO 원본 이미지 개수: {len(presigned_data)}")
    bentoml_logger.info(f"MinIO 원본 이미지 경로: {bucket}/{prefix}")
    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600, sock_connect=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [download_file(session, url, path) for url, path in presigned_data]
        for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="진행률", unit="file"):
            local_path, error = await future
            if error:
                bentoml_logger.info(f"MinIO 이미지 다운로드 실패: {error}")
    print()

#async def upload_to_minio_parallel(source_dir: str, target_prefix: str):
#    loop = asyncio.get_running_loop()
#    files = os.listdir(source_dir)
#    async def upload_file(filename):
#        try:
#            async with aiofiles.open(os.path.join(source_dir, filename), 'rb') as f:
#                content = await f.read()
#            data_stream = BytesIO(content)
#            await loop.run_in_executor(
#                executor,
#                lambda: client.put_object(
#                    bucket,
#                    f"{target_prefix}/{filename}",
#                    data_stream,
#                    length=len(content),
#                    content_type="application/octet-stream"
#                )
#            )
#        except Exception as e:
#            bentoml_logger.error(f"MinIO로 데이터 업로드 중 오류 발생: {e}")
#            raise BentoMLException("MinIO로 데이터 업로드 중 오류가 발생했습니다.")
#    await asyncio.gather(*[upload_file(f) for f in files])

@svc.api(input=JSON(), output=JSON(), route='/preprocessing')
async def preprocessing(params: dict):
    bentoml_logger.info(f"{params}")
    try:
        parsed_params = PreprocessingParams(**params)
    except Exception as e:
        bentoml_logger.error(f"입력 파라미터 파싱 실패: {e}")
        raise BentoMLException("입력값이 유효하지 않습니다.")

    arg_dataPath = [", ".join(parsed_params.dataPath)]
    arg_dataNormalization = [", ".join(parsed_params.dataNormalization)]
    arg_dataAugmentation = [", ".join(parsed_params.dataAugmentation)]

    df = pd.DataFrame({
        'projectId': [parsed_params.projectId],
        'versionId': [parsed_params.versionId],
        'dataPath': [arg_dataPath],
        'dataNormalization': [arg_dataNormalization],
        'dataAugmentation': [arg_dataAugmentation],
        'trainRatio': [parsed_params.trainRatio],
        'validationRatio': [parsed_params.validationRatio],
        'testRatio': [parsed_params.testRatio]
    })

    projectId = parsed_params.projectId
    versionId = parsed_params.versionId
    object_name = f'{projectId}_{versionId}.csv'

    loop = asyncio.get_event_loop()
    def upload():
        with io.StringIO() as csv_buffer:
            df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            with io.BytesIO(csv_bytes) as byte_stream:
                client.put_object(
                    bucket_name=bucket,
                    object_name=object_name,
                    data=byte_stream,
                    length=len(csv_bytes),
                    content_type='application/csv'
                )
    await loop.run_in_executor(executor, upload)

@svc.api(input=JSON(), output=JSON(), route='/datasource')
async def datasource(params: DataSourceParams):
    try:
        projectId = params[0]['projectId']
        versionId = params[0]['versionId']

        key = f"{projectId}_{versionId}"

        csv_save_dir = os.path.join(base_path, projectId, versionId)
        os.makedirs(csv_save_dir, exist_ok=True)
        params_all_df = pd.DataFrame(params)
        params_all_df.to_csv(os.path.join(csv_save_dir, 'datasourceparams.csv'), index=False)

        images_paths = os.path.join(base_path, projectId, 'rawdata', 'images')
        annotations_paths = os.path.join(base_path, projectId, 'rawdata', 'annotations')

        def reset_folder(paths):
            try:
                if os.path.exists(paths):
                    shutil.rmtree(paths)
                os.makedirs(paths)
            except OSError as e:
                bentoml_logger.error(f"이미지 & 어노테이션 폴더 생성 실패: {e}")
                raise BentoMLException("이미지 & 어노테이션 폴더 생성에 실패했습니다.")

        for folder in [images_paths, annotations_paths]:
            reset_folder(folder)

        def indent(elem, level=0):
            i = '\n' + level * '  '
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + '  '
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for child in elem:
                    indent(child, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

        def safe_str(val):
            if val is None:
                return ''
            if isinstance(val, float) and math.isnan(val):
                return ''
            return str(val)                    

        def annotation_information(group_df):
            first_row = group_df.iloc[0]
            root = ET.Element('annotation')
            ET.SubElement(root, 'folder').text = safe_str(first_row['folder'])
            ET.SubElement(root, 'filename').text = safe_str(first_row['filename'])
            ET.SubElement(root, 'path').text = safe_str(first_row['path'])
            source = ET.SubElement(root, 'source')
            ET.SubElement(source, 'database').text = 'GRIT-Dlabflow'
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = safe_str(first_row['width'])
            ET.SubElement(size, 'height').text = safe_str(first_row['height'])
            ET.SubElement(size, 'depth').text = safe_str(first_row['depth'])
            ET.SubElement(root, 'segmented').text = safe_str(first_row['segmented'])
            for _, obj_data in group_df.iterrows():
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = safe_str(obj_data['name'])
                ET.SubElement(obj, 'pose').text = safe_str(obj_data['pose'])
                ET.SubElement(obj, 'truncated').text = safe_str(obj_data.get('truncated', ''))
                ET.SubElement(obj, 'difficult').text = safe_str(obj_data['difficult'])
                ET.SubElement(obj, 'occluded').text = safe_str(obj_data['obj_id'])
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = safe_str(obj_data['xmin'])
                ET.SubElement(bndbox, 'xmax').text = safe_str(obj_data['xmax'])
                ET.SubElement(bndbox, 'ymin').text = safe_str(obj_data['ymin'])
                ET.SubElement(bndbox, 'ymax').text = safe_str(obj_data['ymax'])
            indent(root)
            tree = ET.ElementTree(root)
            xml_filename = os.path.splitext(first_row['filename'])[0] + '.xml'
            xml_path = os.path.join(annotations_paths, xml_filename)
            try:
                tree.write(xml_path)
            except Exception as e:
                bentoml_logger.error(f"XML 파일 생성 실패: {e}")
                raise BentoMLException("XML 파일 생성에 실패했습니다.")

        async def create_datasource_param(item):
            loop = asyncio.get_event_loop()
            def validate():
                try:
                    result = DataSourceParams(**item).dict()
                    return result
                except Exception as e:
                    bentoml_logger.error(f"입력 데이터 유효성 검사 실패: {e}")
                    raise BentoMLException("입력 데이터가 올바르지 않습니다.")
            return await loop.run_in_executor(executor, validate)

        async def convert_to_dataframe_async(arg_dict):
            tasks = [create_datasource_param(item) for item in arg_dict]
            values = await asyncio.gather(*tasks)
            return pd.DataFrame(values)

        try:
            df = await convert_to_dataframe_async(params)
        except Exception as e:
            bentoml_logger.error(f"라벨링 정보 데이터프레임 변환 실패: {e}")
            raise BentoMLException("라벨링 정보를 데이터프레임으로 변환하는 중 오류가 발생했습니다.")

        async def fetch_mysql(query):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: db_mysql_dataframe(query))

        query_stat = f"""
            SELECT * FROM Stat WHERE projectId = '{projectId}' AND versionId = '{versionId}'
        """
        query_preprocessing = f"""
            SELECT * FROM Preprocessing WHERE projectId = '{projectId}' AND versionId = '{versionId}'
        """

        try:
            df_mysql_select_1, df_mysql_select_2 = await asyncio.gather(fetch_mysql(query_stat), fetch_mysql(query_preprocessing))
            totalpages = df['totalPages'].iloc[0]
            groups = df['filename'].nunique()

            cumulative_csv_path = os.path.join(base_path, projectId, versionId, f"{projectId}_{versionId}_cumulative.csv")
            os.makedirs(os.path.dirname(cumulative_csv_path), exist_ok=True)

            statusofdatasource = 'FINISH'
            current_page = 1
            if totalpages > 1:
                if 'READY' in df_mysql_select_1['statusOfDataSource'].values:
                    statusofdatasource = 'RUNNING'
                    current_page = 1
                else:
                    last_page = df_mysql_select_2['PageN'].values[0]
                    next_page = last_page + 1
                    if next_page >= totalpages:
                        current_page = totalpages
                        statusofdatasource = 'FINISH'
                    else:
                        current_page = next_page
                        statusofdatasource = 'RUNNING'

            bentoml_logger.info(f"{current_page}/{totalpages} {statusofdatasource} {groups}")

            if current_page == 1:
                df.to_csv(cumulative_csv_path, index=False, mode='w', encoding='utf-8-sig')
            else:
                if os.path.exists(cumulative_csv_path):
                    df.to_csv(cumulative_csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')
                else:
                    df.to_csv(cumulative_csv_path, index=False, mode='w', encoding='utf-8-sig')

            db_mysql_update(
                sql_select='Stat',
                updates={'statusOfDataSource': statusofdatasource},
                conditions={'projectId': projectId, 'versionId': versionId}
            )
            db_mysql_update(
                sql_select='Preprocessing',
                updates={'PageN': current_page, 'Pagelast': totalpages},
                conditions={'projectId': projectId, 'versionId': versionId}
            )

            if statusofdatasource == 'FINISH':
                cumulative_df = pd.read_csv(cumulative_csv_path)

                loop = asyncio.get_event_loop()
                def generate(group_df):
                    annotation_information(group_df)

                tasks = [loop.run_in_executor(executor, generate, group_df) for _, group_df in cumulative_df.groupby('filename')]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        bentoml_logger.error(f"XML 파일 생성 중 오류 발생: {result}")
                        raise BentoMLException("XML 파일 생성 중 오류가 발생했습니다.")

                bentoml_logger.info(f"Preprocessing part ===> projectId: {projectId}, versionId: {versionId}")
#                name1 = cumulative_df['folder'][0].split('/')[0]
#                name2 = cumulative_df['folder'][0].split('/')[1]
#                await download_from_minio_parallel(name1, name2, target_dir=images_paths)
                unique_folders = cumulative_df['folder'].unique()
                """
                for folder in unique_folders:
                    parts = folder.split('/')
                    if len(parts) < 2:
                        bentoml_logger.error(f"폴더 형식 오류: {folder}")
                        continue
                    name1, name2 = parts[0], parts[1]
                    await download_from_minio_parallel(name1, name2, target_dir=images_paths)
                """
                for folder in unique_folders:
                    if not folder:
                        bentoml_logger.error(f"폴더 형식 오류: {folder}")
                        continue
                    parts = folder.split('/')
                    bucket_name = parts[0]
                    prefix = '/'.join(parts[1:])
                    await download_from_minio_parallel(bucket_name, prefix, target_dir=images_paths)


                object_name = f'{projectId}_{versionId}.csv'
                try:
                    response = client.get_object(bucket, object_name)
                    with response:
                        csv_data = response.read().decode('utf-8')
                except Exception as e:
                    bentoml_logger.error(f"MinIO에서 데이터 전처리와 관련된 변수 다운로드 실패: {e}")
                    raise BentoMLException("MinIO에서 데이터 전처리와 관련된 변수를 다운로드하는데 실패했습니다.")

                minio_df = pd.read_csv(io.StringIO(csv_data))

                db_mysql_update(
                    sql_select='Stat',
                    updates={'statusOfProject': 'RUNNING'},
                    conditions={'projectId': projectId, 'versionId': versionId}
                )

                cmd = (
                    f"python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/preprocessing.py "
                    f"--projectId={minio_df['projectId'].values[0]} "
                    f"--versionId={minio_df['versionId'].values[0]} "
                    f"--dataPath={minio_df['dataPath'].values[0]} "
                    f"--dataNormalization={minio_df['dataNormalization'].values[0]} "
                    f"--dataAugmentation={minio_df['dataAugmentation'].values[0]} "
                    f"--trainRatio={minio_df['trainRatio'].values[0]} "
                    f"--validationRatio={minio_df['validationRatio'].values[0]} "
                    f"--testRatio={minio_df['testRatio'].values[0]}"
                )
                proc = await asyncio.create_subprocess_exec(
                    *shlex.split(cmd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    bentoml_logger.error(f"데이터 전처리 컨테이너 생성 실패: {stderr.decode()}")
                    db_mysql_update(
                        sql_select='Stat',
                        updates={'statusOfProject': 'ERROR'},
                        conditions={'projectId': projectId, 'versionId': versionId}
                    )
                    raise BentoMLException("데이터 전처리 컨테이너를 생성하지 못했습니다.")
                else:
                    bentoml_logger.info(f"데이터 전처리 컨테이너 생성 완료")
                    db_mysql_update(
                        sql_select='Stat',
                        updates={'statusOfProject': 'FINISH'},
                        conditions={'projectId': projectId, 'versionId': versionId}
                    )
        except Exception as e:
            statusofdatasource = 'ERROR'
            bentoml_logger.error(f"MySQL 테이블 조회 실패: {e}")
            raise BentoMLException("MySQL 테이블 조회에 실패했습니다.")
    except BentoMLException:
        raise
    except Exception as e:
        statusofdatasource = 'ERROR'
        bentoml_logger.error(f"서버 내부 오류 발생: {e}")
        raise BentoMLException("서버 내부 오류가 발생했습니다.")

input_spec_training = JSON(pydantic_model=ModelTunningParams)

def background_training_task(arg_dict):
    arg = ModelTunningParams(**arg_dict)
    if arg.algorithm in ('yolo_version_5_normal', 'yolo_version_5_small', 'yolo_version_5_medium', 'yolo_version_5_large', 'yolo_version_5_xlarge', 'yolo_version_8_normal', 'yolo_version_8_small', 'yolo_version_8_medium', 'yolo_version_8_large', 'yolo_version_8_xlarge'):
        if str(arg.tuning).lower() == 'true':
            adv_setting = json.dumps(arg.advancedSettingForObjectDetection.dict()) if arg.advancedSettingForObjectDetection else ""
            cmd = [
                "python3",
                "/mnt/dlabflow/backend/kubeflow/pipelines/admin/training_yolo.py",
                "--projectId", arg.projectId,
                "--versionId", arg.versionId,
                "--algorithm", arg.algorithm,
                "--batchsize", str(arg.batchsize),
                "--epoch", str(arg.epoch),
                "--tuning", str(arg.tuning),
                "--advancedSettingForObjectDetection", adv_setting
            ]
            subprocess.run(cmd, check=True)
        else:
            os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/training_yolo.py --projectId=%s --versionId=%s --algorithm=%s --batchsize=%s --epoch=%s --tuning=%s --advancedSettingForObjectDetection=%s' % (arg.projectId, arg.versionId, arg.algorithm, arg.batchsize, arg.epoch, arg.tuning, arg.advancedSettingForObjectDetection))
    
    elif arg.algorithm in ('efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2', 'efficientdet_d3', 'efficientdet_d4', 'efficientdet_d5', 'efficientdet_d6', 'efficientdet_d7'):
        df = pd.DataFrame([{'epoch': arg.epoch}])
        df_path = '/mnt/dlabflow/backend/kubeflow/pipelines/admin'
        df_file_path = os.path.join(df_path, 'epoch.txt')
        df.to_csv(df_file_path, index=False)
        if str(arg.tuning).lower() == 'true':
            adv_setting = json.dumps(arg.advancedSettingForObjectDetection.dict()) if arg.advancedSettingForObjectDetection else ""
            cmd = [
                "python3",
                "/mnt/dlabflow/backend/kubeflow/pipelines/admin/training_tf.py",
                "--projectId", arg.projectId,
                "--versionId", arg.versionId,
                "--algorithm", arg.algorithm,
                "--batchsize", str(arg.batchsize),
                "--epoch", str(arg.epoch),
                "--tuning", str(arg.tuning),
                "--advancedSettingForObjectDetection", adv_setting
            ]
            subprocess.run(cmd, check=True)
        else:
            os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/training_tf.py --projectId=%s --versionId=%s --algorithm=%s --batchsize=%s --epoch=%s --tuning=%s --advancedSettingForObjectDetection=%s' % (arg.projectId, arg.versionId, arg.algorithm, arg.batchsize, arg.epoch, arg.tuning, arg.advancedSettingForObjectDetection))

@svc.api(input=input_spec_training, output=JSON(), route='/training')
def training(arg: ModelTunningParams):
    print('training')
    print(arg)
    p = Process(target=background_training_task, args=(arg.dict(),))
    p.start()
    return {"status": "accepted", "message": "Training task is being processed in the background", "projectId": arg.projectId, "versionId": arg.versionId}

input_spec_inference = JSON(pydantic_model=InferenceParams)

def background_inference_task(arg_dict):
    arg = InferenceParams(**arg_dict)
    object_path = f"{arg.projectId}/{arg.versionId}/train/algorithm.txt"
    client = Minio('10.40.217.236:9002', 'dlab-backend', 'dlab-backend-secret', secure=False)
    response = client.get_object(bucket, object_path)
    algorithm = response.read().decode('utf-8')
    if algorithm in ('efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2', 'efficientdet_d3', 'efficientdet_d4', 'efficientdet_d5', 'efficientdet_d6', 'efficientdet_d7'):
        os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/inference_tf.py --projectId=%s --versionId=%s --sessionId=%s' % (arg.projectId, arg.versionId, arg.sessionId))
    else:
        os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/inference_yolo.py --projectId=%s --versionId=%s --sessionId=%s' % (arg.projectId, arg.versionId, arg.sessionId))

@svc.api(input=input_spec_inference, output=JSON(), route='/inference')
def inference(arg: InferenceParams):
    print('inference')
    p = Process(target=background_inference_task, args=(arg.dict(),))
    p.start()
    return {"status": "accepted", "message": "Inference task is being processed in the background", "projectId": arg.projectId, "versionId": arg.versionId}

