import os
from functools import partial
from kfp.components import create_component_from_func
from kfp import dsl, onprem
import argparse
import kfp
import requests
from dotenv import load_dotenv, dotenv_values
from kubernetes.client import V1EnvVar
import json

dotenv_path = '/mnt/dlabflow/structured/config'
load_dotenv(dotenv_path)

KubeflowPipelineAutoannotation = os.getenv('KubeflowPipelineAutoannotation')
KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')
KubeflowVolumeName = os.getenv('KubeflowVolumeName1')
KubeflowVolumeMountPath = os.getenv('KubeflowVolumeMountPath1')
KubeflowGPUName = os.getenv('KubeflowGPUName1')
KubeflowGPUValue = os.getenv('KubeflowGPUValue1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:unstructured-objectdetection-20251031-beta')
def Autoannotation(id: int, projectId: str, versionId: str, datasetId: str, autoAlgorithm: str, minConfidence: float, maxConfidence: float, targetImagePaths: list, classDefinitions: list, status: str, statusMessage: str, autoAnnotationResult: str):
    import os
    import requests
    from dotenv import load_dotenv, dotenv_values
    import redis
    import traceback
    import logging
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom.minidom import parseString
    from ultralytics import YOLO
    import shutil
    import pymysql
    from minio import Minio
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
    MinIOBucketPreprocessing = os.getenv('MinIOBucketPreprocessing')
    MinIOBucketTraining = os.getenv('MinIOBucketTraining')
    MinIOBucketAutoannotation = os.getenv('MinIOBucketAutoannotation')
    BentomlAutoannotationGet = os.getenv('BentomlAutoannotationGet')

    base_paths = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinIOBucketAutoannotation}"
    autoannotation_paths = f"{base_paths}/{datasetId}"
    #model_paths = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinIOBucketTraining}/model_weights"
    model_paths = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/train/model/train/weights"
    model_paths_efficiendet = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/train/train/model"
    #pipeline_config_paths = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinIOBucketTraining}/pipeline.config"
    pipeline_config_paths_efficiendet = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/train/train/model/pipeline.config"
    #labelmap_paths = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinIOBucketTraining}/labelmap.pbtxt"
    labelmap_paths_efficiendet = f"{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/train/images/labelmap.pbtxt"
    os.makedirs(base_paths, exist_ok=True)
    os.makedirs(autoannotation_paths, exist_ok=True)

    client = Minio(
        endpoint=MinioEndpoint,
        access_key=MinioAccessKey,
        secret_key=MinioSecretKey,
        secure=MinioSecure
    )

    def send_status_to_bentoml(projectId, versionId, status, statusMessage=None, autoAnnotationResult=None, progress=None):
        try:
            url = BentomlAutoannotationGet
            payload = {
                'projectId': projectId,
                'versionId': versionId,
                'status': status,
            }
            if statusMessage:
                payload['statusMessage'] = statusMessage
            if autoAnnotationResult is not None:
                payload['autoAnnotationResult'] = autoAnnotationResult
            if progress is not None:
                payload['progress'] = progress
            requests.post(url, json=payload, timeout=3)
        except Exception:
            statusMessage = '오토어노테이션 결과를 전송하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
            raise

    def cleanup_folder_contents(path):
        os.makedirs(path, exist_ok=True)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    def cleanup_images_after_annotation(path):
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        for f in os.listdir(path):
            if f.lower().endswith(supported_exts):
                os.remove(os.path.join(path, f))

    def target_image_downloads(targetImagePaths):
        try:
            for path in targetImagePaths:
                bucket, object_name = path.split('/', 1)
                filename = os.path.basename(object_name)
                local_path = os.path.join(autoannotation_paths, filename)
                if os.path.exists(local_path):
                    if os.path.isdir(local_path):
                        shutil.rmtree(local_path)
                    else:
                        os.remove(local_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                client.fget_object(bucket, object_name, local_path)
        except Exception:
            statusMessage = '오토어노테이션 대상 이미지를 MinIO에서 다운로드하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
            raise

    def create_yolo_xml(image_filename, image_path, width, height, depth, detections, save_path, projectId, versionId, datasetId):
        try:
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

            try:
                object_name = f"{projectId}/{versionId}/{MinIOBucketAutoannotation}/{datasetId}/{os.path.basename(xml_filename)}".lstrip('/')
                client.fput_object(bucket_name=MinioBucketUser, object_name=object_name, file_path=xml_filename)
            except Exception as e:
                statusMessage = '오토어노테이션 결과를 MinIO에 업로드하는 과정에서 오류가 발생하였습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
                raise

        except Exception:
            statusMessage = 'XML 파일을 생성하는 과정에서 오류가 발생하였습니다.'
            logger.error(statusMessage)
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
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
                object_name = f"{projectId}/{versionId}/{MinIOBucketAutoannotation}/{datasetId}/{os.path.basename(xml_filename)}".lstrip('/')
                client.fput_object(bucket_name=MinioBucketUser, object_name=object_name, file_path=xml_filename)
            except Exception as e:
                statusMessage = '오토어노테이션 결과를 MinIO에 업로드하는 과정에서 오류가 발생하였습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
                raise

        except Exception:
            statusMessage = 'XML 파일을 생성하는 과정에서 오류가 발생하였습니다.'
            logger.error(statusMessage)
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
            raise        

    def autoannotation_yolo(model_path, image_path, confidence_min, confidence_max, target_classes):
        model = YOLO(model=model_path)
        name_to_id = {v.lower(): k for k, v in model.names.items()}
        target_class_list = [cls.get('className','').strip().lower() for cls in classDefinitions if 'className' in cls]
        class_ids = [name_to_id[name] for name in target_class_list if name in name_to_id]
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(supported_exts)]

        if not image_files:
            statusMessage = '오토어노테이션에 사용할 이미지가 존재하지 않습니다.'
            logger.error(statusMessage)
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
            raise

        results = model.predict(
            image_path,
            classes=class_ids,
            save=False,
            line_width=1,
            name=image_path,
            exist_ok=True,
            conf=confidence_min,
            verbose=True
        )

        for result in results:
            orig_img = result.orig_img
            image_file_path = getattr(result, 'path', 'unknown_path.jpg')
            width, height, depth = orig_img.shape[1], orig_img.shape[0], orig_img.shape[2]
            detections = []

            for box in result.boxes:
                conf = float(box.conf[0])
                if confidence_min < conf < confidence_max:
                    bbox = box.xyxy[0]
                    cls = int(box.cls[0])
                    class_name = result.names[cls].lower()
                    if class_name in target_classes.split(','):
                        detections.append({
                            'xmin': bbox[0],
                            'ymin': bbox[1],
                            'xmax': bbox[2],
                            'ymax': bbox[3],
                            'class_name': class_name
                        })

            if len(detections) == 0:
                logger.info(f"{image_file_path}에서 대상 클래스가 감지되지 않아 오토어노테이션 파일 생성을 생략합니다.")
                continue

            create_yolo_xml(
                image_filename=image_file_path,
                image_path=image_file_path,
                width=width, 
                height=height, 
                depth=depth,
                detections=detections,
                save_path=autoannotation_paths,
                projectId=projectId,
                versionId=versionId,
                datasetId=datasetId
            )

    def autoannotation_efficientdet(model_path, image_path, confidence_min, confidence_max, target_classes):
        target_class_list = [cls.get('className','').strip().lower() for cls in classDefinitions if 'className' in cls]
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(supported_exts)]

        if not image_files:
            statusMessage = '오토어노테이션에 사용할 이미지가 존재하지 않습니다.'
            logger.error(statusMessage)
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
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

            """
            if enable_visualization:
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=minConfidence,
                    agnostic_mode=False
                )

                cv2.imwrite(os.path.join(autoannotation_paths, os.path.basename(image_path)), image_np)
            """

            if len(detections['detection_classes']) == 0:
                logger.info(f"{image_path}에서 대상 클래스가 감지되지 않아 오토어노테이션 파일 생성을 생략합니다.")
                continue

            create_efficientdet_xml(image_path, detections, category_index, autoannotation_paths)

    """ Task """
    logger.info(f"projectId: {projectId}")
    logger.info(f"versionId: {versionId}")
    target_classes = ','.join([cls.get('className', '') for cls in classDefinitions if 'className' in cls])
    send_status_to_bentoml(projectId, versionId, status='RUNNING', statusMessage='오토어노테이션 진행 중입니다.')
    cleanup_folder_contents(autoannotation_paths)
    target_image_downloads(targetImagePaths)
    logger.info(f"{model_paths}")

    if 'yolo' in autoAlgorithm:
        autoannotation_yolo(
            model_path=f"{model_paths}/best.pt",
            image_path=autoannotation_paths,
            confidence_min=minConfidence,
            confidence_max=maxConfidence,
            target_classes=target_classes
        )
    elif 'efficientdet' in autoAlgorithm:
        autoannotation_efficientdet(
            model_path=model_paths_efficiendet,
            image_path=autoannotation_paths,
            confidence_min=minConfidence,
            confidence_max=maxConfidence,
            target_classes=target_classes
        )
    else:
        statusMessage = '오토어노테이션 모델이 존재하지 않습니다.'
        logger.error(statusMessage)
        send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
        raise

    cleanup_images_after_annotation(autoannotation_paths)
    result_path = f"{projectId}/{versionId}/autoannotation/{datasetId}/"
    result_path_list = client.list_objects(MinioBucketUser, prefix=result_path, recursive=True)
    result_files = [f"{MinioBucketUser}/{obj.object_name}" for obj in result_path_list if not obj.object_name.endswith('/')]

    if not os.listdir(autoannotation_paths):
        statusMessage = '오토어노테이션이 완료되었지만, 탐지된 객체가 없어 XML 파일이 생성되지 않았습니다.'
        logger.error(statusMessage)
        send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
        raise

    send_status_to_bentoml(projectId, versionId, status='FINISH', statusMessage='오토어노테이션 완료', autoAnnotationResult=result_files)

def pipelines():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int)
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--datasetId', type=str)
    parser.add_argument('--autoAlgorithm', type=str)
    parser.add_argument('--minConfidence', type=float)
    parser.add_argument('--maxConfidence', type=float)
    parser.add_argument('--targetImagePaths', type=str)
    parser.add_argument('--classDefinitions', type=str)
    parser.add_argument('--status', type=str)
    parser.add_argument('--statusMessage', type=str)
    parser.add_argument('--autoAnnotationResult', type=str)
    args = parser.parse_args()

    target_image_paths = json.loads(args.targetImagePaths) if args.targetImagePaths.startswith('[') else [args.targetImagePaths]

    shm_vol = dsl.PipelineVolume(name='shm-vol', empty_dir={'medium': 'Memory'})
    Autoannotation_task = Autoannotation(args.id, args.projectId, args.versionId, args.datasetId, args.autoAlgorithm, args.minConfidence, args.maxConfidence, target_image_paths, args.classDefinitions, args.status, args.statusMessage, args.autoAnnotationResult) \
     .set_display_name('Model Autoannotation') \
     .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
     .add_env_variable(V1EnvVar(name=f"{KubeflowGPUName}", value=f"{KubeflowGPUValue}")) \
     .add_pvolumes({'/dev/shm': shm_vol}) \
     .execution_options.caching_strategy.max_cache_staleness='P0D'

if __name__ == '__main__':
    pipeline_package_path = f"{KubeflowPipelineAutoannotation}_pipelines.zip"
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
    experiment = client.create_experiment(name=f"{KubeflowPipelineAutoannotation}")
    run = client.run_pipeline(experiment.id, f"{KubeflowPipelineAutoannotation} pipelines", pipeline_package_path)

