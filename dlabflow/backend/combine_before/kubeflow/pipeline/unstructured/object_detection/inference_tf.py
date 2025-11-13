from functools import partial
from kfp.components import create_component_from_func
import kfp
from kfp import onprem
from kfp import compiler
from kfp import dsl
from kfp.dsl import component
import argparse
import requests
import asyncio
import bentoml
from pydantic import BaseModel
import typing as t
import requests
from kubernetes.client import V1EnvVar
from dotenv import load_dotenv, dotenv_values
import os

dotenv_path = '/mnt/dlabflow/backend/kubeflow/config'
load_dotenv(dotenv_path)

KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')
KubeflowVolumeName = os.getenv('KubeflowVolumeName1')
KubeflowVolumeMountPath = os.getenv('KubeflowVolumeMountPath1')
KubeflowPieplineName = os.getenv('KubeflowPieplineInference')
KubeflowGPUName = os.getenv('KubeflowGPUName4')
KubeflowGPUValue = os.getenv('KubeflowGPUValue4')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:unstructured-objectdetection-20251031-beta')
def Inference(projectId: str, versionId: str, sessionId: str):
    import random
    import os
    import glob
    import pandas as pd
    import xml.etree.ElementTree as ET
    import git
    import wget
    import subprocess
    import re
    import tarfile
    import shutil
    import pathlib
    import matplotlib
    import matplotlib.pyplot as plt
    import io
    import scipy.misc
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np; print('numpy version: ', np.__version__)
    from six import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    from six.moves.urllib.request import urlopen
    import tensorflow as tf; print('tensorflow version: ', tf.__version__)
    import tensorflow as tf_hub; print('tensorflow hub version: ', tf_hub.__version__)
    import keras; print('keras version: ', keras.__version__)
    from tensorboard import notebook
    from minio import Minio
    import pymysql
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.utils import ops as utils_opsz    
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
    minio_path = f'/mnt/dlabflow/backend/minio/{bucket}'
    inference_before_path = f'{minio_path}/{projectId}/{versionId}/inference/{sessionId}/before'
    inference_after_path = f'{minio_path}/{projectId}/{versionId}/inference/{sessionId}/after'
    inference_model_poth = f'{minio_path}/{projectId}/{versionId}/train/train'
    result_path = minio_path+'/'+projectId+'/'+versionId+'/train'
    PATH_TO_SAVED_MODEL = inference_model_poth+'/model/save/saved_model'
    os.makedirs(inference_before_path, exist_ok=True)
    os.makedirs(inference_after_path, exist_ok=True)

    ################################################################################################
    ## model task 1 : predict
    ################################################################################################

    client = Minio(endpoint=MinioEndpoint, access_key=MinioAccessKey, secret_key=MinioSecretKey, secure=MinioSecure)

    for item in client.list_objects(bucket, prefix=f'{projectId}/{versionId}/inference/{sessionId}/before', recursive=True):
        client.fget_object(bucket, item.object_name, f'{minio_path}/{item.object_name}')

    def predict():
        try:
            detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")            

        try:
            label_map_path = result_path + '/images/labelmap.pbtxt'
            category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
            print("Label map loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load label map: {e}")        

        image_files = [f for f in os.listdir(inference_before_path) if f.endswith('.jpg') or f.endswith('.png')]
        if not image_files:
            raise FileNotFoundError("No image files found for inference.")

        print(f"Found {len(image_files)} images for inference.")

        for image_file in image_files:
            try:
                image_path = os.path.join(inference_before_path, image_file)
                print(f"Processing: {image_path}")
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis,...]
                detections = detect_fn(input_tensor)
                print(f"Detections keys: {detections.keys()}")
                boxes = detections['detection_boxes'][0].numpy()
                classes = detections['detection_classes'][0].numpy().astype(np.int32)
                scores = detections['detection_scores'][0].numpy()
                threshold = 0.5
                valid_detections = sum(score >= threshold for score in scores)
                if valid_detections > 0:
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        boxes,
                        classes,
                        scores,
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4,
                        min_score_thresh=threshold,
                    )
                else:
                    print(f"No valid detections found for {image_file}. Saving image without boxes.")

                result_image_path = os.path.join(inference_after_path, image_file)
                result_image = Image.fromarray(image_np)
                result_image.save(result_image_path)
                print(f"Saved: {result_image_path}")
                for file in os.listdir(inference_after_path):
                    client.fput_object(bucket, f'{projectId}/{versionId}/inference/{sessionId}/after/{file}', f'{inference_after_path}/{file}')
            except Exception as e:
                print(f"Error inference {image_file}: {e}")

    ################################################################################################
    ## preprocessing task 1 run
    ################################################################################################

    def update_inference_status(status: str):
        conn = pymysql.connect(host=PyMySQLHost, user=PyMySQLUser, password=PyMySQLPassword, port=PyMySQLPort, db=PyMySQLDB, charset='utf8')
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE Inference SET statusOfInference=%s WHERE projectId=%s AND versionId=%s"
                cursor.execute(sql, (status, projectId, versionId))
            conn.commit()
        finally:
            conn.close()

    try:
        update_inference_status('RUNNING')
        predict()
        update_inference_status('FINISH')
    except Exception as e:
        print(f'[ERROR] Inference failed: {e}')
        update_inference_status('ERROR')
    finally:
        cleanup_keys = [
            f'{projectId}_{versionId}.csv',
            f'{projectId}/{versionId}/inference/{sessionId}/before/.keep',
            f'{projectId}/{versionId}/inference/{sessionId}/after/.keep'
        ]
        for key in cleanup_keys:
            try:
                client.remove_object(bucket, key)
            except Exception as e:
                print(f'[WARN] Failed to remove {key}: {e}')            

################################################################################################
## kubeflow pipeline upload
################################################################################################

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--sessionId', type=str)
    args = parser.parse_args()

    Inference_apply = Inference(args.projectId, args.versionId, args.sessionId) \
        .set_display_name('Model Inference') \
        .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
        .add_env_variable(V1EnvVar(name=f"{KubeflowGPUName}", value=f"{KubeflowGPUValue}"))

    smh_vol = kfp.dsl.PipelineVolume(name = 'shm-vol', empty_dir = {'medium': 'Memory'})
    Inference_apply.add_pvolumes({'/dev/shm': smh_vol})        
    Inference_apply.execution_options.caching_strategy.max_cache_staleness = 'P0D'

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
