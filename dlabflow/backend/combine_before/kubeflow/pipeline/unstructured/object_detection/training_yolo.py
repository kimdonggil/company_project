from functools import partial
from kfp.components import create_component_from_func
import kfp
from kfp import onprem
from kfp import compiler
from kfp import dsl
from kfp.dsl import component
from typing import Optional, List
import argparse
import requests
import asyncio
import bentoml
from pydantic import BaseModel
import json
import typing as t
import requests
from kubernetes.client import V1EnvVar
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
KubeflowPieplineName = os.getenv('KubeflowPieplineTraining')
KubeflowGPUName = os.getenv('KubeflowGPUName4')
KubeflowGPUValue = os.getenv('KubeflowGPUValue4')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:unstructured-objectdetection-20251031-beta')
def Training(projectId: str, versionId: str, algorithm: str, batchsize: int, epoch: int, tuning: str, advancedSettingForObjectDetection: Optional[str] = None):
    from ultralytics import YOLO
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import shutil
    import xml.etree.ElementTree as ET
    import decimal
    import os
    import json
    import datetime
    import glob
    from glob import glob
    import tqdm
    from typing import Optional, List
    import random
    import math
    from minio import Minio
    import bentoml
    import csv
    from pathlib import Path
    import pymysql
    import sys
    import time
    from datetime import datetime, timedelta
    import logging
    import torch
    import subprocess
    from io import BytesIO
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
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    minio_path = '/mnt/dlabflow/backend/minio/'+bucket
    preprocessing_path = minio_path+'/'+projectId+'/'+versionId+'/preprocessing'
    result_path = minio_path+'/'+projectId+'/'+versionId+'/train'
    training_executed_path = result_path+f"/app/{projectId}_{versionId}.txt"
    client = Minio(endpoint=MinioEndpoint, access_key=MinioAccessKey, secret_key=MinioSecretKey, secure=MinioSecure)
    db = pymysql.connect(host=PyMySQLHost, user=PyMySQLUser, password=PyMySQLPassword, port=PyMySQLPort, db=PyMySQLDB, charset='utf8')

    def db_update(table, set_dict, projectId, versionId):
        cursor = db.cursor()
        try:
            set_clause = ', '.join(f"{k}=%s" for k in set_dict)
            sql = f"UPDATE {table} SET {set_clause} WHERE projectId=%s AND versionId=%s"
            val = list(set_dict.values()) + [projectId, versionId]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()

    def sample(data):
        image_list = sorted([f for f in os.listdir(data) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
        files = []
        if len(image_list) == 10:
            for i in range(len(random.sample(image_list, 10))):
                f = data+'/'+image_list[i]
                files.append(f)
        else:
            for i in range(len(random.sample(image_list, len(image_list)))):
                f = data+'/'+image_list[i]
                files.append(f)
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(11.69, 8.27), tight_layout=True)
        axes = axes.flatten()
        for idx, (ax, file) in enumerate(zip(axes, files)):
            pic = plt.imread(file)
            ax.imshow(pic)
            ax.axis('off')
        else:
            [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]
        fig.savefig(result_path+'/validation_result.jpg', dpi=300)

    def upload_text_to_minio(bucket_name, object_path, text):
        data = text.encode('utf-8')
        data_stream = BytesIO(data)
        client.put_object(bucket_name=bucket_name, object_name=object_path, data=data_stream, length=len(data), content_type='text/plain')

    def yolo():
        def load_annotations(preprocessing_path):
            annotation_files = [os.path.join(root, f) for root, _, files in os.walk(preprocessing_path) for f in files if f.endswith('.xml')]
            data = []
            class_set = set()
            for ann_file in annotation_files:
                tree = ET.parse(ann_file)
                root = tree.getroot()
                filename = root.find('filename').text
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    class_set.add(label)
                    bndbox = obj.find('bndbox')
                    bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                    data.append([filename, label, bbox, width, height])
            class_lists = sorted(list(class_set))
            class_id = {name: idx for idx, name in enumerate(class_lists)}
            df = pd.DataFrame(data, columns=['filename', 'label', 'bboxes', 'width', 'height'])
            df['class_id'] = df['label'].map(class_id)
            df['bboxes_str'] = df['bboxes'].apply(lambda x: ','.join(map(str, x)))
            df = df.drop_duplicates(subset=['filename','class_id','bboxes_str'])
            df = df.drop(columns=['bboxes_str'])
            return df, class_id

        def pascal_voc_to_yolo_bbox(bbox_array, w, h):
            x_min, y_min, x_max, y_max = bbox_array
            x_center = ((x_max + x_min) / 2) / w
            y_center = ((y_max + y_min) / 2) / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h
            return [x_center, y_center, width, height]

        def create_yolo_dataset(split_name, dataset_path, result_path, df_data):
            src_path = os.path.join(f"{dataset_path}/datasplit", split_name)
            dst_path = os.path.join(result_path, 'data', split_name)
            os.makedirs(dst_path, exist_ok=True)
            valid_extensions = ['jpg', 'png', 'JPG', 'PNG']
            image_list = [f for f in os.listdir(src_path) if f.split('.')[-1] in valid_extensions]
            for image in image_list:
                file_stem = Path(image).stem
                df_image = df_data[df_data['filename'].apply(lambda x: Path(x).stem) == file_stem]
                if df_image.empty:
                    continue
                txt_file = os.path.join(dst_path, file_stem + '.txt')
                with open(txt_file, 'w') as f:
                    for idx, row in df_image.iterrows():
                        yolo_bbox = pascal_voc_to_yolo_bbox(row['bboxes'], row['width'], row['height'])
                        line = str(row['class_id']) + " " + " ".join(map(str, yolo_bbox))
                        f.write(line + '\n')
                shutil.copy2(os.path.join(src_path, image), dst_path)

        df_data, class_id = load_annotations(preprocessing_path)
        classes = list(class_id.keys())
        class_count = len(classes)
        for split in ['train', 'val', 'test']:
            create_yolo_dataset(split, preprocessing_path, result_path, df_data)

        """
        for i in os.listdir(train_path):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/data/train/'+i, file_path=train_path+'/'+i)
        for i in os.listdir(val_path):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/data/val/'+i, file_path=val_path+'/'+i)
        for i in os.listdir(test_path):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/data/test/'+i, file_path=test_path+'/'+i)
        """

        train_path = result_path+'/data/train'
        val_path = result_path+'/data/val'
        test_path = result_path+'/data/test'

        yaml = f"""
            train: {train_path}
            val: {val_path}
            test: {test_path}
            nc: {class_count}
            names: {classes}
            """
        
        with open(result_path+'/custom.yaml', 'w') as f:
            f.write(yaml)

        """
        yaml_file = [file for file in os.listdir(result_path) if file.endswith('.yaml')]
        for i in yaml_file:
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)
        """

        cuda = torch.cuda.is_available()
        if cuda == True:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
            class CustomCallback:
                def __init__(self, epoch):
                    self.epoch = epoch
                    self.start_time = datetime.now()
                def __call__(self, model, *args, **kwargs):
                    try:
                        current_time = datetime.now().strftime('%H:%M:%S')
                        epoch = getattr(model, 'epoch', 'unknown')
                        if epoch != 'unknown':
                            progress_percentage = (epoch / self.epoch) * 100
                            db_update('Training', {'trainProgress': progress_percentage, 'epoch': epoch}, projectId, versionId)
                        else:
                            logging.info(f"Epoch information is not available. Progress: unknown% at {current_time}.")
                    except Exception as e:
                        logging.error(f"Error in CustomCallback: {e}")
                        raise

            model_files = {
                'yolo_version_5_normal': 'yolov5nu.pt',
                'yolo_version_5_small': 'yolov5su.pt',
                'yolo_version_5_medium': 'yolov5mu.pt',
                'yolo_version_5_large': 'yolov5lu.pt',
                'yolo_version_5_xlarge': 'yolov5xu.pt',
                'yolo_version_8_normal': 'yolov8n.pt',
                'yolo_version_8_small': 'yolov8s.pt',
                'yolo_version_8_medium': 'yolov8m.pt',
                'yolo_version_8_large': 'yolov8l.pt',
                'yolo_version_8_xlarge': 'yolov8x.pt',
            }

            model_file = model_files.get(algorithm)
            if model_file:
                model = YOLO(model_file)

            custom_callback = CustomCallback(epoch=epoch)
            model.add_callback('on_train_epoch_end', custom_callback)
            val_images = glob(os.path.join(val_path, '**', '*.*'), recursive=True)
            val_images = [img for img in val_images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if len(val_images) <= 0:
                trainingerrorcategory = 4
                error_message = "검증 데이터가 없습니다. 최소 1개 이상으로 버전 생성을 다시 해주세요.\n"
                with open(result_path + "/error.log", "w") as f:
                    f.write(error_message)
                raise ValueError(error_message)

            def get_ratio(epochs, ratio):
                return max(1, int(epochs * ratio))

            def train_settings():
                for key, value in train_args.items():
                    print(f"{key}: {value}")

            train_args = {
                'data': result_path+'/custom.yaml',
                'epochs': epoch,
                'batch': batchsize,
                'project': result_path+'/model',
                'exist_ok': True,
                'device': 1
            }

            if tuning.lower() == 'false':
                train_args.update({
                    'patience': epoch
                })
            else:
                tuningOption_dict = json.loads(advancedSettingForObjectDetection)
                patiences = get_ratio(epoch, tuningOption_dict['patience'])
                close_mosaics = get_ratio(epoch, tuningOption_dict['closeMosaic'])
                warmup_epochs_float = epoch*tuningOption_dict['warmupEpochs']
                if warmup_epochs_float.is_integer():
                    warmup_epochs_int = int(warmup_epochs_float)
                else:
                    warmup_epochs_int = round(warmup_epochs_float)
                train_args.update({
                    'imgsz': tuningOption_dict['imgSize'],
                    'lr0': tuningOption_dict['lr0'],
                    'lrf': tuningOption_dict['lrf'],
                    'cos_lr': tuningOption_dict['cosLr'],
                    'warmup_epochs': warmup_epochs_int,
                    'warmup_bias_lr': tuningOption_dict['warmupBiasLr'],
                    'optimizer': tuningOption_dict['optimizer'],
                    'momentum': tuningOption_dict['momentum'],
                    'warmup_momentum': tuningOption_dict['warmupMomentum'],
                    'weight_decay': tuningOption_dict['weightDecay'],
                    'patience': patiences,
                    'freeze': tuningOption_dict['freeze'],
                    'multi_scale': tuningOption_dict['multiScale'],
                    'amp': tuningOption_dict['amp'],
                    'close_mosaic': close_mosaics,
                    'dropout': tuningOption_dict['dropout'],
                    'box': tuningOption_dict['box'],
                    'cls': tuningOption_dict['cls']
                })

            train_settings()

            model.train(**train_args)

            df = pd.read_csv(result_path+'/model/train/results.csv')
            last_epoch = df.iloc[-1]
            score_dict = {
                'precision': last_epoch['metrics/precision(B)'],
                'recall': last_epoch['metrics/recall(B)'],
                'mAP50': last_epoch['metrics/mAP50(B)'],
            }

            values = []
            for k, v in score_dict.items():
                values.append(['{}'.format(k), '{:.4f}'.format(v)])

            values_pd = pd.DataFrame(values, columns=['metric', 'value'])
            values_pd.to_csv(result_path+'/metrics_score.csv', index=False)

            metrics_score = [file for file in os.listdir(result_path) if file.endswith('metrics_score.csv')]
            for i in metrics_score:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)
            best_pt = [file for file in os.listdir(result_path+'/model/train/weights') if file.endswith('best.pt')]
            for i in best_pt:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/model/train/weight/'+i, file_path=result_path+'/model/train/weights/'+i)

            def create_zip(input_path, output_path, zip_name="model_weight", max_size=10, split_size="2M"):
                files = []
                for root, dirs, file_names in os.walk(input_path):
                    for file_name in file_names:
                        file_path = os.path.join(root, file_name)
                        files.append(file_path)
                zip_file_base = os.path.join(output_path, zip_name)
                total_size = sum(os.path.getsize(f) for f in files)
                total_size_MB = total_size / (1024 * 1024)
                if total_size_MB <= max_size:
                    command = ["7z", "a", f"{zip_file_base}.7z", *files]
                    subprocess.run(command, check=True)
                else:
                    command = ["7z", "a", f"{zip_file_base}.7z", *files, f"-v{split_size}"]
                    subprocess.run(command, check=True)

            input_path = result_path+'/model/train/weights'
            output_path = result_path+'/model/train/weights_zip'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            create_zip(input_path, output_path, zip_name="model_weight", max_size=1024, split_size="1024M")

            zip_files = [file for file in os.listdir(output_path) if file.startswith('model_weight')]
            for i in zip_files:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/model_weight/'+i, file_path=output_path+'/'+i)

            df = pd.read_csv(result_path+'/model/train/results.csv', header=0)
            df.columns = df.columns.str.strip()
            fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
            ax.plot(df['epoch'].values, df['train/box_loss'].values, '.-', label='Train Loss', color='b')
            ax.plot(df['epoch'].values, df['val/box_loss'].values, '.-', label='Validation Loss', color='r')
            ax.set_title('Bounding Box Loss', fontsize=20, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=15, fontweight='bold')
            ax.set_ylabel('Loss (Pred. - True)', fontsize=15, fontweight='bold')            
            ax.grid()
            ax.patch.set_facecolor('white')
            ax.legend(loc=0)
            ax.grid(axis='x')
            ax.grid(axis='y')
            plt.savefig(result_path+'/model/train/metrics_chart.jpg', transparent = True)
            metrics_chart = [file for file in os.listdir(result_path+'/model/train') if file.endswith('metrics_chart.jpg')]
            for i in metrics_chart:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/model/train/'+i)
            validation_result = model.predict(source=val_path, save=True, project=result_path, name='result', exist_ok=True)
            sample(result_path+'/result')
            validation_results = [file for file in os.listdir(result_path) if file.endswith('validation_result.jpg')]
            for i in validation_results:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)

            db_update('Training', {'trainProgress': 100, 'epoch': epoch}, projectId, versionId)

            algorithm_path = f"{projectId}/{versionId}/train/algorithm.txt"
            upload_text_to_minio(bucket, algorithm_path, algorithm)     

        else:
            print('GPU is not using')

    try:
        db_update('Stat', {'statusOfTrain': 'RUNNING'}, projectId, versionId)
        db_update('Training', {'subStatusOfTraining': 'RUNNING'}, projectId, versionId)
        db_update('Error', {'TrainingErrorCategory': 0, 'TrainingErrorLog': '정상'}, projectId, versionId)
        yolo()
        metrics_file = f"{projectId}/{versionId}/train/metrics_score.csv"
        for item in client.list_objects(bucket_name=bucket, prefix=f"{projectId}/{versionId}/train", recursive=True):
            if item.object_name == metrics_file:
                metrics = client.get_object(bucket, item.object_name)
                df_metrics = pd.read_csv(metrics, index_col=0, names=['metrics', 'value'], header=0)
                break
        precision = df_metrics['value'][0]
        recall = df_metrics['value'][1]
        mAP = df_metrics['value'][2]
        db_update('Training', {'algorithm': algorithm, 'batchsize': batchsize, 'mAP': mAP, 'recall': recall, 'precisions': precision, 'subStatusOfTraining': 'FINISH'}, projectId, versionId)

    except Exception as e:
        db_update('Stat', {'statusOfTrain': 'ERROR'}, projectId, versionId)
        db_update('Training', {'subStatusOfTraining': 'ERROR'}, projectId, versionId)
        error_message = f"Error: {e}\n"
        with open(result_path+"/error.log", "w") as f:
            f.write(error_message)
        if os.path.exists(result_path+"/error.log"):
            with open(result_path+"/error.log", "r") as f:
                error_content = f.read()
            if "CUDA out of memory" in error_content:
                trainingerrorcategory = 1
                #new_content = "GPU 메모리 부족으로 학습이 중단되면서 오류가 발생하였습니다."
            elif "root" in error_content or "broken permissions" in error_content:
                trainingerrorcategory = 2
                #new_content = "컨테이너 연결 과정에서 root 권한이 부여되지 않아 오류가 발생하였습니다."
            elif "BadRequest" in error_content or "PodInitializing" in error_content:
                trainingerrorcategory = 3
                #new_content = "컨테이너를 생성하지 못하여 오류가 발생하였습니다."
                #new_content = "알 수 없는 오류"
            elif "검증 데이터가 10개 미만입니다" in error_content:
                trainingerrorcategory = 4

        db_update('Error', {'TrainingErrorCategory': trainingerrorcategory, 'TrainingErrorLog': error_message}, projectId, versionId)
        print(f"Error: {e}")
        raise

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--tuning', type=str)
    parser.add_argument('--advancedSettingForObjectDetection', type=str)
    args = parser.parse_args()
    Training_apply = Training(args.projectId, args.versionId, args.algorithm, args.batchsize, args.epoch, args.tuning, args.advancedSettingForObjectDetection) \
        .set_display_name('Model Training') \
        .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
        .add_env_variable(V1EnvVar(name=f"{KubeflowGPUName}", value=f"{KubeflowGPUValue}"))
    
    smh_vol = kfp.dsl.PipelineVolume(name = 'shm-vol', empty_dir = {'medium': 'Memory'})
    Training_apply.add_pvolumes({'/dev/shm': smh_vol})        
    Training_apply.execution_options.caching_strategy.max_cache_staleness = 'P0D'

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
