import asyncio
import io
import shlex
import json
import sys
import logging
import os
import shlex
import warnings
from contextlib import contextmanager
from datetime import timedelta
from io import BytesIO
from typing import List, Optional
import xml.etree.ElementTree as ET
import aiofiles
import aiohttp
import pandas as pd
import pymysql
from concurrent.futures import ThreadPoolExecutor
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, validator
from tqdm.asyncio import tqdm_asyncio
import bentoml
from bentoml.exceptions import BentoMLException
from bentoml.io import JSON
warnings.filterwarnings("ignore")

# ------------------- Logger Init -------------------

def init_logger(name="bentoml", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
bentoml_logger = init_logger()

# ------------------- Error Helper -------------------

def log_and_raise_error(message: str, exc: Exception):
    bentoml_logger.error(f"{message}: {exc}")
    raise BentoMLException(message)

# ------------------- Progress -------------------

def print_progress(msg, completed, total, width=40):
    progress_str = f"{msg} 진행률: {completed}/{total}"
    bentoml_logger.info(progress_str)

def log_info(msg, completed=None, total=None):
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()
    bentoml_logger.info(msg)
    if completed is not None and total is not None:
        print_progress(completed, total)

# ------------------- Constants & Globals -------------------

bucket = 'grit'
minio_path = '/mnt/dlabflow/backend/minio'
base_path = os.path.join(minio_path, bucket)
client = Minio('10.40.217.236:9002', 'dlab-backend', 'dlab-backend-secret', secure=False)
executor = ThreadPoolExecutor(max_workers=20)
svc = bentoml.Service('autoannotation')

# ------------------- Models -------------------

class ClassDefinition(BaseModel):
    className: str
    imageDatasetAnnotationTechnique: str

class AutoannotationParams(BaseModel):
    id: int
    projectId: str
    versionId: str
    datasetId: str
    autoAlgorithm: str
    minConfidence: float
    maxConfidence: float
    targetImagePaths: List[str]
    classDefinitions: List[ClassDefinition]

    @validator('targetImagePaths', pre=True)
    def convert_to_dir_paths(cls, v):
        from pathlib import Path
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception as e:
                raise ValueError(f"targetImagePaths가 JSON 형식 문자열이 아님: {e}")
        if not isinstance(v, list):
            raise ValueError("targetImagePaths는 리스트여야 합니다")
        return list(set(str(Path(p).parent) for p in v))

    @validator('classDefinitions', pre=True)
    def parse_class_definitions(cls, v):
        if not isinstance(v, list):
            v = [v]
        new_list = []
        for i, item in enumerate(v):
            if isinstance(item, dict):
                new_list.append(item)
            elif isinstance(item, str):
                new_list.append({
                    "className": item,
                    "imageDatasetAnnotationTechnique": "default"
                })
            else:
                raise ValueError(f"classDefinitions element {i} is not dict or str but {type(item)}")
        return new_list

# ------------------- MySQL -------------------

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
        return pd.read_sql(sql_select, db)

def db_mysql_update(sql_select, updates: dict, conditions: dict):
    with mysql_connection() as (db, cursor):
        set_clause = ', '.join([f"{k}=%s" for k in updates.keys()])
        where_clause = ' AND '.join([f"{k}=%s" for k in conditions.keys()])
        sql = f"UPDATE {sql_select} SET {set_clause} WHERE {where_clause}"
        values = tuple(updates.values()) + tuple(conditions.values())
        try:
            cursor.execute(sql, values)
            db.commit()
        except Exception as e:
            db.rollback()
            log_and_raise_error(f"{sql_select} 테이블 업데이트 실패", e)

# ------------------- MinIO -------------------

async def download_from_minio_parallel(bucket: str, prefix: str, target_dir: str, target_files: Optional[List[str]] = None):
    os.makedirs(target_dir, exist_ok=True)
    loop = asyncio.get_running_loop()
    def get_presigned_url(object_name):
        return client.presigned_get_object(bucket, object_name, expires=timedelta(seconds=3600))
    async def download_file(session, url, local_path):
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            async with session.get(url) as response:
                if response.status != 200:
                    return local_path, f"HTTP 오류: {response.status}"
                async with aiofiles.open(local_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024 * 64):
                        await f.write(chunk)
            return local_path, None
        except Exception as e:
            return local_path, str(e)    
    async def prepare_download_list():
        objects = await loop.run_in_executor(executor, lambda: list(client.list_objects(bucket, prefix=prefix, recursive=True)))
        presigned_data = []
        for obj in objects:
            relative_path = os.path.relpath(obj.object_name, prefix)
            if target_files is not None and relative_path not in target_files:
                continue
            url = await loop.run_in_executor(executor, get_presigned_url, obj.object_name)
            local_path = os.path.join(target_dir, relative_path)
            presigned_data.append((url, local_path))
        return presigned_data
    presigned_data = await prepare_download_list()
    if not presigned_data:
        error_message = "MinIO에서 오토어노테이션에 사용할 이미지가 없습니다. 최소 1개 이상의 이미지가 필요합니다.\n"
        bentoml_logger.error(error_message)
        return 0
    bentoml_logger.info(f"MinIO 오토어노테이션 이미지 개수: {len(presigned_data)}")
    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600, sock_connect=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [download_file(session, url, path) for url, path in presigned_data]
        total = len(tasks)
        completed = 0
        for future in asyncio.as_completed(tasks):
            local_path, error = await future
            completed += 1
            print_progress('다운로드', completed, total)
            if error:
                log_info(f"MinIO 이미지 다운로드 실패: {local_path}, 오류: {error}", completed, total)
        return len(presigned_data)

async def fetch_mysql(query):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: db_mysql_dataframe(query))

# ------------------- Main BentoML API -------------------

@svc.api(input=JSON(), output=JSON(), route='/autoannotation')
async def autoannotation(params: dict):
    try:
        parsed_params = AutoannotationParams(**params)
        bentoml_logger.info(f"오토어노테이션 요청 수신: {params}")
        id = parsed_params.id
        projectId = parsed_params.projectId
        versionId = parsed_params.versionId
        datasetId = parsed_params.datasetId
        autoAlgorithm = parsed_params.autoAlgorithm
        minConfidence = parsed_params.minConfidence
        maxConfidence = parsed_params.maxConfidence
        targetImagePaths = parsed_params.targetImagePaths
        classDefinitions = parsed_params.classDefinitions
        targetimages = params['targetImagePaths']
        targetimageslist = [os.path.basename(path) for path in targetimages]

        query = f"SELECT * FROM Autoannotations WHERE datasetId = '{datasetId}'"
        df = (await asyncio.gather(fetch_mysql(query)))[0]
        if df['statusOfAutoAnnotation'].iloc[-1] == 'READY':
            bentoml_logger.info(f"오토어노테이션 시작 - projectId: {projectId}, versionId: {versionId}, datasetId: {datasetId}")
            images_paths = os.path.join(base_path, projectId, versionId, datasetId)
            db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': 'RUNNING'}, {'id': id, 'datasetId': datasetId})
            try:
                path = targetImagePaths[0]
                name1, name2 = path.split('/', 1)
                download_count = await download_from_minio_parallel(name1, name2, target_dir=images_paths, target_files=targetimageslist)
                if download_count == 0:
                    bentoml_logger.error('MinIO에서 다운로드할 오토어노테이션 이미지가 없습니다. 오토어노테이션 스크립트 실행을 건너뜁니다.')
                    db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': 'FINISH', 'resultLabelingPaths': '[]'}, {'id': id, 'datasetId': datasetId})
                    return {"message": "No images to process. Autoannotation skipped."}
                current_row = df[df['id'] == id].iloc[0]
                current_paths = json.loads(current_row['targetImagePaths'])
                prev_df = df[(df['id'] < id) & (df['datasetId'] == datasetId)].sort_values(by="id", ascending=False)
                if not prev_df.empty:
                    prev_row = prev_df.iloc[0]
                    prev_paths = json.loads(prev_row['targetImagePaths'])
                    if (sorted(current_paths) == sorted(prev_paths) and current_row['resultLabelingPaths'] in ('[]', None, '', 'null')):
                        bentoml_logger.warning('이전 요청과 동일한 이미지이며 생성된 XML 파일이 없습니다. 오토어노테이션 스크립트 실행을 건너뜁니다.')
                        db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': 'FINISH', 'resultLabelingPaths': '[]'}, {'id': id, 'datasetId': datasetId})
                        return {"message": "Same images with no xml file. Autoannotation skipped."}
            except Exception as e:
                db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': 'ERROR'}, {'id': id, 'datasetId': datasetId})
                log_and_raise_error("오토어노테이션 스크립트 실행 중 오류 발생", e)
            target_image_paths_json = json.dumps(targetImagePaths)
            class_definitions_json = json.dumps([cls.dict() for cls in classDefinitions])
            cmd = (
                f"python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/autoannotation.py "
                f"--id={id} "
                f"--projectId={projectId} "
                f"--versionId={versionId} "
                f"--datasetId={datasetId} "
                f"--autoAlgorithm={autoAlgorithm} "
                f"--minConfidence={minConfidence} "
                f"--maxConfidence={maxConfidence} "
                f"--targetImagePaths={shlex.quote(target_image_paths_json)} "
                f"--classDefinitions={shlex.quote(class_definitions_json)}"
            )
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': 'ERROR'}, {'id': id, 'datasetId': datasetId})
                log_and_raise_error("오토어노테이션 스크립트 실행 실패", stderr.decode())
            bentoml_logger.info("오토어노테이션 스크립트 실행 완료")
    except Exception as e:
        projectId = params.get("projectId", "UNKNOWN")
        versionId = params.get("versionId", "UNKNOWN")
        try:
            db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': 'ERROR'}, {'id': id, 'datasetId': datasetId})
        except Exception as db_e:
            bentoml_logger.error(f"DB 상태 업데이트 실패: {db_e}")
        log_and_raise_error("오토어노테이션 API 처리 실패", e)
