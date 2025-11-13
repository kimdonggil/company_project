from functools import partial
from kfp.components import create_component_from_func
from kfp import compiler, dsl, onprem
from kubernetes.client import V1EnvVar
import argparse
import json
import requests
import kfp
import logging
import os
from dotenv import load_dotenv, dotenv_values

dotenv_path = '/mnt/dlabflow/backend/kubeflow/config'
load_dotenv(dotenv_path)

KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')
KubeflowVolumeName = os.getenv('KubeflowVolumeName1')
KubeflowVolumeMountPath = os.getenv('KubeflowVolumeMountPath1')
KubeflowPieplineName = os.getenv('MinIOBucketAutoannotation')
KubeflowGPUName = os.getenv('KubeflowGPUName1')
KubeflowGPUValue = os.getenv('KubeflowGPUValue1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:unstructured-objectdetection-20251031-beta')
def Autoannotation(id: int, projectId: str, versionId: str, datasetId: str, autoAlgorithm: str, minConfidence: float, maxConfidence: float, targetImagePaths: list, classDefinitions: list):
    import os
    from concurrent.futures import ThreadPoolExecutor
    import traceback
    import logging
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom.minidom import parseString
    from ultralytics import YOLO
    import shutil
    from minio import Minio
    import pymysql
    from contextlib import contextmanager
    from typing import List, Dict
    import glob
    import requests
    import tarfile
    from pathlib import Path
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xml.etree.ElementTree as ET
    from xml.dom import minidom    
    import cv2
    import tensorflow as tf
    from google.protobuf import text_format
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from object_detection.utils import config_util
    from object_detection.builders import model_builder
    from object_detection.protos import pipeline_pb2
    from object_detection.model_lib_v2 import train_loop
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils       
    from dotenv import load_dotenv, dotenv_values

    dotenv_path = '/mnt/dlabflow/backend/kubeflow/config'
    load_dotenv(dotenv_path)

    PyMySQLHost = os.getenv('PyMySQLHost')
    PyMySQLUser = os.getenv('PyMySQLUser')
    PyMySQLPassword = os.getenv('PyMySQLPassword')
    PyMySQLPort = int(os.getenv('PyMySQLPort1'))
    PyMySQLDB = os.getenv('PyMySQLDB1')
    MinIODefaultPath = os.getenv('MinIODefaultPath')
    MinioEndpoint = os.getenv('MinioEndpoint')
    MinioAccessKey = os.getenv('MinioAccessKey')
    MinioSecretKey = os.getenv('MinioSecretKey')
    MinioSecure = os.getenv('MinioSecure', 'False').lower() == 'true'
    bucket = os.getenv('MinioBucketUser1')    

    MinioBucketUser = bucket
    save_path = f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/{datasetId}"
    model_paths = f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/train/train/model"
    autoannotation_paths = f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/{datasetId}"
    pipeline_config_paths_efficiendet = f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/train/train/model/pipeline.config"
    labelmap_paths_efficiendet = f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/train/images/labelmap.pbtxt"

    client = Minio(endpoint=MinioEndpoint, access_key=MinioAccessKey, secret_key=MinioSecretKey, secure=MinioSecure)

    logger = logging.getLogger("Autoannotation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @contextmanager
    def mysql_connection():
        db = pymysql.connect(host=PyMySQLHost, user=PyMySQLUser, password=PyMySQLPassword, port=PyMySQLPort, db=PyMySQLDB, charset='utf8')
        cursor = db.cursor()
        try:
            yield db, cursor
        finally:
            cursor.close()
            db.close()

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

    def batch_update_autoannotations(data: list[dict], batch_size: int = 500):
        if not data:
            return
        with mysql_connection() as (db, cursor):
            try:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    status_case = " ".join("WHEN id = %s AND datasetId = %s THEN %s" for _ in batch)
                    path_case = " ".join("WHEN id = %s AND datasetId = %s THEN %s" for _ in batch)
                    where_clause = " OR ".join("(id = %s AND datasetId = %s)" for _ in batch)
                    sql = f"""
                        UPDATE Autoannotations
                        SET
                            statusOfAutoAnnotation = CASE {status_case} END,
                            resultLabelingPaths = CASE {path_case} END
                        WHERE {where_clause}
                    """
                    values = []
                    for d in batch:
                        values.extend([d['id'], d['datasetId'], d['statusOfAutoAnnotation']])
                    for d in batch:
                        values.extend([d['id'], d['datasetId'], d['resultLabelingPaths']])
                    for d in batch:
                        values.extend([d['id'], d['datasetId']])
                    cursor.execute(sql, values)
                db.commit()
            except Exception as e:
                db.rollback()
                log_and_raise_error("오토어노테이션 테이블 업데이트 실패", e)

    def create_xml(image_filename, image_path, width, height, depth, detections, save_path, bucket, projectId, versionId, datasetId):
        annotation = Element('annotation')
        folder = SubElement(annotation, 'folder')
#        folder.text = os.path.dirname(image_path)
#        folder.text = f"{bucket}/{projectId}/{versionId}/{datasetId}"
        folder.text = f"{targetImagePaths[0]}"
        filename = SubElement(annotation, 'filename')
        filename.text = os.path.basename(image_path)
        path = SubElement(annotation, 'path')
#        path.text = image_path
#        path.text = f"{bucket}/{projectId}/{versionId}/{datasetId}/{os.path.basename(image_path)}"
        path.text = f"{targetImagePaths[0]}/{filename.text}"
        source = SubElement(annotation, 'source')
        database = SubElement(source, 'database')
        database.text = 'dlabflow'
        size = SubElement(annotation, 'size')
        SubElement(size, 'width').text = str(width)
        SubElement(size, 'height').text = str(height)
        SubElement(size, 'depth').text = str(depth)
        SubElement(annotation, 'segmented').text = '0'
        for detection in detections:
            obj = SubElement(annotation, 'object')
            SubElement(obj, 'name').text = detection['class_name']
            SubElement(obj, 'pose').text = 'Unspecified'
            SubElement(obj, 'truncated').text = '0'
            SubElement(obj, 'difficult').text = '0'
            SubElement(obj, 'occluded').text = '0'
            bndbox = SubElement(obj, 'bndbox')
            SubElement(bndbox, 'xmin').text = str(int(detection['xmin']))
            SubElement(bndbox, 'ymin').text = str(int(detection['ymin']))
            SubElement(bndbox, 'xmax').text = str(int(detection['xmax']))
            SubElement(bndbox, 'ymax').text = str(int(detection['ymax']))
        xml_str = tostring(annotation)
        pretty_xml = parseString(xml_str).toprettyxml(indent="  ")
        xml_filename = os.path.join(save_path, os.path.splitext(os.path.basename(image_filename))[0] + '.xml')
        with open(xml_filename, 'w') as f:
            f.write(pretty_xml)

    def create_xml_wrapper(args):
        try:
            create_xml(**args)
        except Exception as e:
            logger.error(f"XML 생성 실패 (병렬): {e}")

    def autoannotation(model_path, image_path, predict_path, annotation_path, confidence_min, confidence_max, target_classes):
        model = YOLO(model=model_path)
        name_to_id = {v.lower(): k for k, v in model.names.items()}
        target_class_list = [cls.strip().lower() for cls in target_classes.split(',')]
        class_ids = [name_to_id[name] for name in target_class_list if name in name_to_id]
        if os.path.isdir(image_path):
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            _ = len([f for f in os.listdir(image_path) if f.lower().endswith(valid_exts)])
        results = model.predict(image_path, classes=class_ids, save=False, line_width=1, name=predict_path, exist_ok=True, conf=confidence_min)
        try:
            if os.path.exists(save_path):        
                shutil.rmtree(save_path)    
            os.makedirs(save_path)            
            xml_tasks = []
            for result in results:
                names = result.names
                orig_img = result.orig_img
                image_file_path = getattr(result, 'path', 'unknown_path.jpg')
                width, height, depth = orig_img.shape[1], orig_img.shape[0], orig_img.shape[2]
                detections = []
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if confidence_min < conf < confidence_max:
                        bbox = box.xyxy[0]
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        detections.append({'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3], 'class_name': class_name})
                if detections:
#                    create_xml(
#                        image_filename=image_file_path,
#                        image_path=image_file_path,
#                        width=width,
#                        height=height,
#                        depth=depth,
#                        detections=detections,
#                        save_path=save_path,
#                        bucket=bucket,
#                        projectId=projectId,
#                        versionId=versionId,
#                        datasetId=datasetId,
#                    )
                    xml_tasks.append({
                        'image_filename': image_file_path,
                        'image_path': image_file_path,
                        'width': width,
                        'height': height,
                        'depth': depth,
                        'detections': detections,
                        'save_path': save_path,
                        'bucket': bucket,
                        'projectId': projectId,
                        'versionId': versionId,
                        'datasetId': datasetId,
                    })
            with ThreadPoolExecutor(max_workers=20) as executor:
                executor.map(create_xml_wrapper, xml_tasks)                    
        except Exception as e:
            logger.error(f"XML 생성 실패: {e}\n{traceback.format_exc()}")
            raise

    def create_efficientdet_xml(image_path, detections, category_index, save_path):
        try:
            image = cv2.imread(image_path)
            height, width, depth = image.shape
            boxes = detections['detection_boxes']
            classes = detections['detection_classes']
            scores = detections['detection_scores']
            annotation = Element('annotation')
            folder = SubElement(annotation, 'folder'); folder.text = f"{targetImagePaths[0]}"
            filename = SubElement(annotation, 'filename'); filename.text = os.path.basename(image_path)
            path = SubElement(annotation, 'path'); path.text = f"{targetImagePaths[0]}/{filename.text}"
            source = SubElement(annotation, 'source'); SubElement(source, 'database').text = 'dlabflow'
            size = SubElement(annotation, 'size')
            SubElement(size, 'width').text = str(width)
            SubElement(size, 'height').text = str(height)
            SubElement(size, 'depth').text = str(depth)
            SubElement(annotation, 'segmented').text = '0'

            for i in range(len(boxes)):
                score = scores[i]
                if score < 0.5:
                    continue
                class_id = int(classes[i])
                if class_id in category_index:
                    class_name = category_index[class_id]['name']
                else:
                    class_name = f"class_{class_id}"
                box = boxes[i]
                ymin = int(box[0] * height)
                xmin = int(box[1] * width)
                ymax = int(box[2] * height)
                xmax = int(box[3] * width)
                obj = SubElement(annotation, 'object')
                SubElement(obj, 'name').text = class_name
                SubElement(obj, 'pose').text = 'Unspecified'
                SubElement(obj, 'truncated').text = '0'
                SubElement(obj, 'difficult').text = '0'
                SubElement(obj, 'occluded').text = '0'
                bndbox = SubElement(obj, 'bndbox')
                SubElement(bndbox, 'xmin').text = str(xmin)
                SubElement(bndbox, 'ymin').text = str(ymin)
                SubElement(bndbox, 'xmax').text = str(xmax)
                SubElement(bndbox, 'ymax').text = str(ymax)

            xml_str = tostring(annotation)
            parsed = minidom.parseString(xml_str)
            pretty_xml = parsed.toprettyxml(indent='  ')
            pretty_xml = pretty_xml.replace('<folder/>', '<folder></folder>')
            pretty_xml_no_header = '\n'.join(pretty_xml.split('\n')[1:])
            xml_filename = os.path.join(save_path, os.path.basename(image_path).rsplit('.', 1)[0] + '.xml')
            with open(xml_filename, 'w', encoding='utf-8') as f:
                f.write(pretty_xml_no_header)

            try:
                object_name = f"{projectId}/{versionId}/{datasetId}/{os.path.basename(xml_filename)}".lstrip('/')
                client.fput_object(bucket_name=MinioBucketUser, object_name=object_name, file_path=xml_filename)
            except Exception as e:
                statusMessage = '오토어노테이션 결과를 MinIO에 업로드하는 과정에서 오류가 발생하였습니다.'
                logger.error(f"{statusMessage}")
                raise

        except Exception:
            statusMessage = 'XML 파일을 생성하는 과정에서 오류가 발생하였습니다.'
            logger.error(statusMessage)
            raise

    def autoannotation_efficientdet(model_path, image_path, confidence_min, confidence_max, target_classes):
        target_class_list = [cls.get('className','').strip().lower() for cls in classDefinitions if 'className' in cls]
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(supported_exts)]

        if not image_files:
            statusMessage = '오토어노테이션에 사용할 이미지가 존재하지 않습니다.'
            logger.error(statusMessage)
            raise

        inference_image_paths = []
        for ext in supported_exts:
            inference_image_paths.extend(glob.glob(os.path.join(autoannotation_paths, f"*{ext}")))

        checkpoint_path = tf.train.latest_checkpoint(model_paths)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                pass

        configs = config_util.get_configs_from_pipeline_file(pipeline_config_paths_efficiendet)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        ckpt = tf.train.Checkpoint(model=detection_model)
        ckpt.restore(checkpoint_path).expect_partial()

        @tf.function(reduce_retracing=True)
        def detect_fn_fixed(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections

        category_index = label_map_util.create_category_index_from_labelmap(labelmap_paths_efficiendet, use_display_name=True)
        resize_shape = (640, 640)

        for image_path in inference_image_paths:
            image_np = cv2.imread(image_path)
            if image_np is None:
                print(f"Cannot read {image_path}, skipping.")
                continue

            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_resized = tf.image.resize(image_rgb, resize_shape)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_resized, 0), dtype=tf.float32)
            detections = detect_fn_fixed(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            scores = detections['detection_scores']
            mask = (scores >= minConfidence) & (scores <= maxConfidence)
            for key in ['detection_boxes', 'detection_classes', 'detection_scores']:
                detections[key] = detections[key][mask]

            indices = [
                i for i, cid in enumerate(detections['detection_classes'])
                if cid in category_index and category_index[cid]['name'] in target_class_list
            ]

            for key in ['detection_boxes', 'detection_classes', 'detection_scores']:
                detections[key] = detections[key][indices]

            if len(detections['detection_classes']) == 0:
                logger.info(f"{image_path}에서 대상 클래스가 감지되지 않아 오토어노테이션 파일 생성을 생략합니다.")
                continue

            create_efficientdet_xml(image_path, detections, category_index, autoannotation_paths)

    def upload_xmls_to_minio(local_dir, bucket, prefix):
#        try:
#            objects_to_delete = [
#                obj.object_name for obj in client.list_objects(bucket, prefix=prefix, recursive=True)
#                if obj.object_name.lower().endswith('.xml')
#            ]
#            for obj_name in objects_to_delete:
#                client.remove_object(bucket, obj_name)
#            logger.info(f"기존 XML 파일 {len(objects_to_delete)}개 삭제 완료: {prefix}")
#        except Exception as e:
#            logger.error(f"기존 MinIO XML 삭제 실패: {e}")
#            raise
#        for fname in os.listdir(local_dir):
#            if fname.endswith('.xml'):
#                file_path = os.path.join(local_dir, fname)
#                try:
#                    with open(file_path, 'rb') as f:
#                        client.put_object(
#                            bucket,
#                            f"{prefix}/{fname}",
#                            f,
#                            length=os.path.getsize(file_path),
#                            content_type='application/octet-stream'
#                        )
#                    logger.info(f"업로드 완료: {prefix}/{fname}")
#                except Exception as e:
#                    logger.error(f"업로드 실패: {fname}, 오류: {e}")

        def upload_file(fname):
            if not fname.endswith('.xml'):
                return
            file_path = os.path.join(local_dir, fname)
            try:
                with open(file_path, 'rb') as f:
                    client.put_object(
                        bucket,
                        f"{prefix}/{fname}",
                        f,
                        length=os.path.getsize(file_path),
                        content_type='application/octet-stream'
                    )
#                logger.info(f"업로드 완료: {prefix}/{fname}")
            except Exception as e:
                logger.error(f"업로드 실패: {fname}, 오류: {e}")
        xml_files = [f for f in os.listdir(local_dir) if f.endswith('.xml')]
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(upload_file, xml_files)

    try:
        logger.info("Autoannotation 시작")
        target_classes = ','.join([cls.get('className', '') for cls in classDefinitions if 'className' in cls])
        logger.info(f"{autoAlgorithm}")
        if 'yolo' in autoAlgorithm:
            autoannotation(
                model_path=f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/train/model/train/weights/best.pt",
                image_path=f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/{datasetId}",
                predict_path=f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/{datasetId}",
                annotation_path=f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/rawdata/annotations",
                confidence_min=minConfidence,
                confidence_max=maxConfidence,
                target_classes=target_classes
            )
        elif 'efficientdet' in autoAlgorithm:
            autoannotation_efficientdet(
                model_path=f"/mnt/dlabflow/backend/minio/{bucket}/{projectId}/{versionId}/train/train/model",
                image_path=autoannotation_paths,
                confidence_min=minConfidence,
                confidence_max=maxConfidence,
                target_classes=target_classes
            )


        upload_xmls_to_minio(
            local_dir=save_path,
            bucket=bucket,
            prefix=f"{projectId}/{versionId}/{datasetId}"
        )
        result_path = f"{projectId}/{versionId}/{datasetId}/"
        result_path_list = client.list_objects(bucket, prefix=result_path, recursive=True)
        result_path_lists = []
        for obj in result_path_list:
            if not obj.object_name.endswith('/'):
                result_path_lists.append(f"{bucket}/{obj.object_name}")
#        logger.info(result_path_lists)
        try:
            save_filenames = set(os.listdir(save_path))
        except FileNotFoundError:
            save_filenames = set()
        result_path_lists_filtered = [path for path in result_path_lists if os.path.basename(path) in save_filenames]
#        logger.info(result_path_lists_filtered)
        if len(result_path_lists) == 0:
            status = 'ERROR'
            resultlabelingpaths = json.dumps([])
        else:
            status = 'FINISH'
            #resultlabelingpaths = json.dumps(result_path_lists)
            resultlabelingpaths = json.dumps(result_path_lists_filtered)
            #logger.info(f"{resultlabelingpaths}")
        #db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': status, 'resultLabelingPaths': resultlabelingpaths}, {'id': id, 'datasetId': datasetId})
        batch_update_autoannotations([{'id': id, 'datasetId': datasetId, 'statusOfAutoAnnotation': status, 'resultLabelingPaths': resultlabelingpaths}])
        logger.info("Autoannotation 완료")
    except Exception as e:
        #db_mysql_update('Autoannotations', {'statusOfAutoAnnotation': 'ERROR', 'resultLabelingPaths': json.dumps([])}, {'id': id, 'datasetId': datasetId})
        batch_update_autoannotations([{'id': id, 'datasetId': datasetId, 'statusOfAutoAnnotation': 'ERROR', 'resultLabelingPaths': json.dumps([])}])
        logger.error(f"Autoannotation 실패: {e}\n{traceback.format_exc()}")
        raise

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', type=int)
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--datasetId', type=str)
    parser.add_argument('--autoAlgorithm', type=str)
    parser.add_argument('--minConfidence', type=float)
    parser.add_argument('--maxConfidence', type=float)
    parser.add_argument('--targetImagePaths', type=str)
    parser.add_argument('--classDefinitions', type=str)
    args = parser.parse_args()

    target_image_paths = json.loads(args.targetImagePaths)
    class_definitions = json.loads(args.classDefinitions)    

    Autoannotation_apply = Autoannotation(
        args.id, args.projectId, args.versionId, args.datasetId, args.autoAlgorithm,
        args.minConfidence, args.maxConfidence,
        json.dumps(target_image_paths),
        json.dumps(class_definitions),
    ) \
    .set_display_name('Model Autoannotation') \
    .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
    .add_env_variable(V1EnvVar(name=f"{KubeflowGPUName}", value=f"{KubeflowGPUValue}"))

    shm_vol = dsl.PipelineVolume(name='shm-vol', empty_dir={'medium': 'Memory'})
    Autoannotation_apply.add_pvolumes({'/dev/shm': shm_vol})
    Autoannotation_apply.execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = f"{KubeflowPieplineName}_pipelines.zip"
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
    experiment = client.create_experiment(name=f"{KubeflowPieplineName}")
    run = client.run_pipeline(experiment.id, f"{KubeflowPieplineName} pipelines", pipeline_package_path)    
