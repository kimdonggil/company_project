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
import json
from pydantic import BaseModel
from typing import Optional, List
import typing as t
import requests
from kubernetes.client import V1EnvVar
import random
import re
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
KubeflowGPUName = os.getenv('KubeflowGPUName1')
KubeflowGPUValue = os.getenv('KubeflowGPUValue1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:unstructured-objectdetection-20251031-beta')
def Training(projectId: str, versionId: str, algorithm:str, batchsize:int, epoch:int, tuning: str, advancedSettingForObjectDetection: Optional[str] = None):
    from ultralytics import YOLO
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import shutil
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import decimal
    import os
    import json
    import datetime
    import glob
    import tqdm
    import re
    import random
    import math
    from minio import Minio
    import bentoml
    import csv
    from pathlib import Path
    import pymysql
    import sys
    import time
    from datetime import datetime
    import logging
    import torch
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
    TensorFlowModels = os.getenv('TensorFlowModels1')
    bucket = os.getenv('MinioBucketUser1')

    minio_path = '/mnt/dlabflow/backend/minio/'+bucket
    preprocessing_path = minio_path+'/'+projectId+'/'+versionId+'/preprocessing'
    client = Minio(endpoint=MinioEndpoint, access_key=MinioAccessKey, secret_key=MinioSecretKey, secure=MinioSecure)
    result_path = minio_path+'/'+projectId+'/'+versionId+'/train'
    training_executed_path = result_path+f"/app/{projectId}_{versionId}.txt"

    tf_path = f"/mnt/dlabflow/backend/kubeflow/pipelines/{TensorFlowModels}/models/research/object_detection"
    df = pd.DataFrame([{
        'base': minio_path,
        'bucket': bucket,
        'projectId': projectId,
        'versionId': versionId
    }])
    paths_file_path = os.path.join(tf_path, 'paths.txt')
    df.to_csv(paths_file_path, index=False, sep='\t')

    epoch_path = f"/mnt/dlabflow/backend/kubeflow/pipelines/{TensorFlowModels}"
    epoch_file_path = os.path.join(epoch_path, 'epoch.txt')
    epoch_df = pd.read_csv(epoch_file_path, sep='\t')
    epoch = epoch_df['epoch'].values[0]

    ################################################################################################
    ## model task 1 : efficientdet
    ################################################################################################

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

    def efficientdet(epoch):

        print('')
        print('+-'*20 + '+')
        print('라이브러리 확인')
        print('')

        import random
        import os
        import glob
        import pandas as pd
        import xml.etree.ElementTree as ET
        import json
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


        print('')
        print('+-'*20 + '+')
        print('TensorFlow Object Detection API 설치')
        print('+-'*20 + '+')        
        print('')        

        if shutil.which('protoc') is None:
            print('protoc가 설치되지 않아 설치를 시작합니다.')
            os.system("""
            cd /tmp
            wget https://github.com/google/protobuf/releases/download/v23.4/protoc-23.4-linux-x86_64.zip
            unzip protoc-23.4-linux-x86_64.zip -d protoc3
            mv protoc3/bin/* /usr/local/bin/
            mv protoc3/include/* /usr/local/include/
            """)
        else:
            print('protoc가 설치되어 있습니다.')

        if shutil.which('cmake') is None:
            print('cmake가 설치되지 않아 설치를 시작합니다.')
            os.system("""
            cd /tmp
            wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6.tar.gz
            tar -xvf cmake-3.27.6.tar.gz
            cd cmake-3.27.6
            ./bootstrap
            make -j $(nproc)
            make install
            """)
        else:
            print('cmake가 설치되어 있습니다.')

        if os.path.exists(f"/mnt/dlabflow/backend/kubeflow/pipelines/{TensorFlowModels}/models"):
            print('TensorFlow Model 폴더가 존재합니다.')
            from object_detection.utils import label_map_util
            from object_detection.utils import visualization_utils as viz_utils
            from object_detection.utils import ops as utils_opsz
        else:
            tensorflowmodels = f"/mnt/dlabflow/backend/kubeflow/pipelines/{TensorFlowModels}"
            print('TensorFlow Model 폴더가 존재하지 않아 설치를 시작합니다.')
            os.system("""
            cd {tensorflowmodels}
            git clone --depth 1 https://github.com/tensorflow/models
            cd models/research/
            protoc object_detection/protos/*.proto --python_out=.
            cp object_detection/packages/tf2/setup.py .
            python -m pip install .
            """)
            from object_detection.utils import label_map_util
            from object_detection.utils import visualization_utils as viz_utils
            from object_detection.utils import ops as utils_ops

        print('')
        print('+-'*20 + '+')
        print('학습 데이터 생성')
        print('+-'*20 + '+')        
        print('')            

        train_path = result_path+'/images/train'
        test_path = result_path+'/images/test'
        path_list = [train_path, test_path]
        for pl in path_list:
            if not os.path.exists(pl):
                os.makedirs(pl, exist_ok=True)

        file2 = preprocessing_path+'/datasplit/train'
        file3 = result_path+'/images/train'
        os.makedirs(file3, exist_ok=True)

        for filename in os.listdir(file2):
            if filename.lower().endswith((".jpg", ".png", ".xml")):
                src_path = os.path.join(file2, filename)
                dst_path = os.path.join(file3, filename)
                shutil.copy2(src_path, dst_path)

        file2 = preprocessing_path+'/datasplit/val'
        file3 = result_path+'/images/test'
        os.makedirs(file3, exist_ok=True)

        for filename in os.listdir(file2):
            if filename.lower().endswith((".jpg", ".png", ".xml")):
                src_path = os.path.join(file2, filename)
                dst_path = os.path.join(file3, filename)
                shutil.copy2(src_path, dst_path)

        file2 = preprocessing_path+'/datasplit/test'
        file3 = result_path+'/images/inference'
        os.makedirs(file3, exist_ok=True)

        for filename in os.listdir(file2):
            if filename.lower().endswith((".jpg", ".png")):
                src_path = os.path.join(file2, filename)
                dst_path = os.path.join(file3, filename)
                shutil.copy2(src_path, dst_path)

        print('학습 데이터가 준비되었습니다.')

        print('')
        print('+-'*20 + '+')
        print('Train & Test Label File')
        print('+-'*20 + '+')        
        print('')                

        def dataframe_csv(annotation_list, annotation_path, name):
            data_dict = {'filename':[], 'width': [], 'height': [], 'class':[], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}
            for i in annotation_list:
                tree = ET.parse(annotation_path+'/'+i)
                root = tree.getroot()
                filename = root.find('filename').text
                for obj in root.findall('object'):
                    classes = obj.find('name').text
                    bndbox_tree = obj.find('bndbox')
                    size = root.find('size')
                    data_dict['filename'].append(filename)
                    data_dict['width'].append(int(size.find('width').text))
                    data_dict['height'].append(int(size.find('height').text))
                    data_dict['class'].append(classes)
                    data_dict['xmin'].append(bndbox_tree.find('xmin').text)
                    data_dict['ymin'].append(bndbox_tree.find('ymin').text)
                    data_dict['xmax'].append(bndbox_tree.find('xmax').text)
                    data_dict['ymax'].append(bndbox_tree.find('ymax').text)
                df_data = pd.DataFrame(data_dict)
                df_data.to_csv(result_path+'/images/'+name+'.csv', index=False)

        train_list = [f for f in os.listdir(train_path) if f.endswith(tuple(['xml']))]
        test_list = [f for f in os.listdir(test_path) if f.endswith(tuple(['xml']))]

        dataframe_csv(train_list, train_path, 'train_labels')
        dataframe_csv(test_list, test_path, 'test_labels')

        classname_path = [result_path+'/images/train_labels.csv', result_path+'/images/test_labels.csv']
        cl_list = []
        for cl in classname_path:
            for cl_ in pd.unique(pd.read_csv(cl)['class']):
                cl_list.append(cl_)
        for cl in list(set(cl_list)):
            print(cl)
        unique_classes = list(set(cl_list))

        print('')
        print('+-'*20 + '+')
        print('Class Name File')
        print('+-'*20 + '+')        
        print('')        

        with open(result_path+'/images/class-names.txt', 'w') as f:
            for i, cl in enumerate(unique_classes):
                if i < len(unique_classes) - 1:
                    f.write(cl + '\n')
                else:
                    f.write(cl)

        print('class-names.txt 파일이 생성되었습니다.')

        print('')
        print('+-'*20 + '+')
        print('Labelmap File')
        print('+-'*20 + '+')        
        print('')

        with open(result_path+'/images/labelmap.pbtxt', 'w') as f:
            items = []
            for idx, cl in enumerate(unique_classes, start=1):
                item_text = (
                    "item {\n"
                    f"  id: {idx}\n"
                    f"  name: '{cl}'\n"
                    "}"
                )
                items.append(item_text)
            
            f.write("\n\n".join(items))

        print('labelmap.pbtxt 파일이 생성되었습니다.')

        print('')
        print('+-'*20 + '+')
        print('TFRecord File')
        print('+-'*20 + '+')        
        print('')

        tf_models_path = f"/mnt/dlabflow/backend/kubeflow/pipelines/{TensorFlowModels}"

        if os.path.exists(tf_models_path+'/models/research/object_detection/tfrecord.py'):
            print('TensorFlow Model 폴더에 tfrecord.py 파일이 존재합니다.')
        else:
            print('TensorFlow Model 폴더에 tfrecord.py 파일이 존재하지 않아 생성합니다.')
            if not os.path.exists(result_path + '/images/object_detection_setting'):
                git.Git(result_path + '/images/').clone('https://github.com/hojihun5516/object_detection_setting.git')
            
            destination_dir = tf_models_path+'/models/research/object_detection/object_detection_setting'
            if os.path.exists(destination_dir):
                shutil.rmtree(destination_dir)

            shutil.move(result_path + '/images/object_detection_setting', destination_dir)
            
            tfrecord_file = tf_models_path+'/models/research/object_detection/object_detection_setting/generate_tfrecord.py'
            with open(tfrecord_file) as f:
                tfrecord = f.read()
            with open(result_path + '/images/tfrecord.py', 'w') as f:
                tfrecord = re.sub(result_path + '/images/class-names.txt', '{}/class-names.txt'.format(result_path + '/images'), tfrecord)
                f.write(tfrecord)
            shutil.move(result_path + '/images/tfrecord.py', tf_models_path+'/models/research/object_detection/')           

        os.system(f"""
        python {tf_models_path}/models/research/object_detection/tfrecord.py \
        --csv_input={result_path}/images/train_labels.csv \
        --output_path={result_path}/images/train.tfrecord \
        --image_dir={result_path}/images/train && \
        python {tf_models_path}/models/research/object_detection/tfrecord.py \
        --csv_input={result_path}/images/test_labels.csv \
        --output_path={result_path}/images/test.tfrecord \
        --image_dir={result_path}/images/test
        """)


        print('')        
        print('+-'*20 + '+')
        print('모델 선택')
        print('+-'*20 + '+')        
        print('')        

        MODELS_CONFIG = {
            'efficientdet_d4_1024x1024': {
                'model_name': 'ssd_efficientdet_d4_1024x1024_coco17_tpu-32',
                'base_pipeline_file': 'ssd_efficientdet_d4_1024x1024_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz',
            },
            'efficientdet_d7_1536x1536': {
                'model_name': 'ssd_efficientdet_d7_1536x1536_coco17_tpu-32',
                'base_pipeline_file': 'ssd_efficientdet_d7_1536x1536_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz',
            },
            'centernet_resnet50_v1_fpn_Keypoints_512x512': {
                'model_name': 'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8',
                'base_pipeline_file': 'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz',
            },
            'faster_rcnn_resnet152_v1_1024x1024': {
                'model_name': 'faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz',
            },
            'resnet152_v1_fpn_1024x1024': {
                'model_name': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8',
                'base_pipeline_file': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
            },
            'resnet50_v1_fpn_1024x1024': {
                'model_name': 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8',
                'base_pipeline_file': 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
            },
            'faster_rcnn_resnet152_v1_640x640': {
                'model_name': 'faster_rcnn_resnet152_v1_640x640_coco17_tpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz',
            },
            'faster_rcnn_resnet50_v1_640x640': {
                'model_name': 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz',
            },
            'efficientdet_d3_896x896': {
                'model_name': 'ssd_efficientdet_d3_896x896_coco17_tpu-32',
                'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz',
            },
            'faster_rcnn_resnet50_v1_1024x1024': {
                'model_name': 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz',
            },
            'efficientdet_d2_768x768': {
                'model_name': 'ssd_efficientdet_d2_768x768_coco17_tpu-8',
                'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz',
            },
            'efficientdet_d6_1408x1408': {
                'model_name': 'ssd_efficientdet_d6_1408x1408_coco17_tpu-32',
                'base_pipeline_file': 'ssd_efficientdet_d6_1408x1408_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz',
            },
            'faster_rcnn_resnet101_v1_800x1333': {
                'model_name': 'faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz',
            },
            'centernet_hourglass104_512x512': {
                'model_name': 'centernet_hourglass104_512x512_coco17_tpu-8',
                'base_pipeline_file': 'centernet_hourglass104_512x512_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz',
            },
            'resnet50_v1_fpn_640x640': {
                'model_name': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
                'base_pipeline_file': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
            },
            'centernet_resnet50_v2_Keypoints_512x512': {
                'model_name': 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8',
                'base_pipeline_file': 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz',
            },
            'mask_rcnn_inception_resnet_v2_1024x1024': {
                'model_name': 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8',
                'base_pipeline_file': 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz',
            },
            'mobilenet_v1_fpn_640x640': {
                'model_name': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
                'base_pipeline_file': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz',
            },
            'resnet152_v1_fpn_640x640': {
                'model_name': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8',
                'base_pipeline_file': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz',
            },
            'centernet_resnet101_v1_fpn_512x512': {
                'model_name': 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
                'base_pipeline_file': 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz',
            },
            'faster_rcnn_resnet101_v1_1024x1024': {
                'model_name': 'faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz',
            },
            'efficientdet_d5_1280x1280': {
                'model_name': 'ssd_efficientdet_d5_1280x1280_coco17_tpu-32',
                'base_pipeline_file': 'ssd_efficientdet_d5_1280x1280_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz',
            },
            'faster_rcnn_resnet152_v1_800x1333': {
                'model_name': 'faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz',
            },
            'mobilenet_v2_fpnlite_320x320': {
                'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
                'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
            },
            'faster_rcnn_resnet101_v1_640x640': {
                'model_name': 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz',
            },
            'centernet_hourglass104_Keypoints_1024x1024_': {
                'model_name': 'centernet_hourglass104_1024x1024_kpts_coco17_tpu-32',
                'base_pipeline_file': 'centernet_hourglass104_1024x1024_kpts_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz',
            },
            'faster_rcnn_resnet50_v1_800x1333': {
                'model_name': 'faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8',
                'base_pipeline_file': 'faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz',
            },
            'centernet_hourglass104_1024x1024': {
                'model_name': 'centernet_hourglass104_1024x1024_coco17_tpu-32',
                'base_pipeline_file': 'centernet_hourglass104_1024x1024_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz',
            },
            'mobilenet_v2_320x320': {
                'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
                'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
            },
            'efficientdet_d0_512x512': {
                'model_name': 'ssd_efficientdet_d0_512x512_coco17_tpu-8',
                'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz',
            },
            'mobilenet_v2_fpnlite_640x640': {
                'model_name': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8',
                'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz',
            },
            'resnet101_v1_fpn_640x640': {
                'model_name': 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8',
                'base_pipeline_file': 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz',
            },
            'efficientdet_d1_640x640': {
                'model_name': 'ssd_efficientdet_d1_640x640_coco17_tpu-8',
                'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz',
            },
            'resnet101_v1_fpn_1024x1024': {
                'model_name': 'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8',
                'base_pipeline_file': 'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
            },
            'centernet_hourglass104_Keypoints_512x512': {
                'model_name': 'centernet_hourglass104_512x512_kpts_coco17_tpu-32',
                'base_pipeline_file': 'centernet_hourglass104_512x512_kpts_coco17_tpu-32.config',
                'pretrained_checkpoint': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz',
            },
        }

        def get_model_config(chosen_model):
            model_config = MODELS_CONFIG.get(chosen_model)
            if model_config:
                return model_config['model_name'], model_config['base_pipeline_file'], model_config['pretrained_checkpoint']
            else:
                raise ValueError(f"Model '{chosen_model}' not found in configuration.")        

        if algorithm == 'efficientdet_d0':
            chosen_model = 'efficientdet_d0_512x512'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)

        elif algorithm == 'efficientdet_d1':
            chosen_model = 'efficientdet_d1_640x640'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)

        elif algorithm == 'efficientdet_d2':
            chosen_model = 'efficientdet_d2_768x768'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)

        elif algorithm == 'efficientdet_d3':
            chosen_model = 'efficientdet_d3_896x896'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)

        elif algorithm == 'efficientdet_d4':
            chosen_model = 'efficientdet_d4_1024x1024'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)

        elif algorithm == 'efficientdet_d5':
            chosen_model = 'efficientdet_d5_1280x1280'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)

        elif algorithm == 'efficientdet_d6':
            chosen_model = 'efficientdet_d6_1408x1408'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)

        elif algorithm == 'efficientdet_d7':
            chosen_model = 'efficientdet_d7_1536x1536'
            model_name, base_pipeline_file, pretrained_checkpoint = get_model_config(chosen_model)
            print('Model Name:', model_name)
            print('Pretrained Checkpoint:', pretrained_checkpoint)
            print('Base Pipeline File:', base_pipeline_file)        

        print('')
        print('+-'*20 + '+')
        print('모델 훈련 조건')
        print('+-'*20 + '+')
        print('')

        num_steps = epoch*100
        batch_size = batchsize

        print('Epoch:', int(num_steps/100))
        print('batch_size:', batch_size)

        print('')
        print('+-'*20 + '+')
        print('모델 훈련 경로')
        print('+-'*20 + '+')
        print('')

        training_path = result_path+'/train/model'
        os.makedirs(training_path, exist_ok=True)
        os.chdir(training_path)

        print('모델 훈련 경로가 생성되었습니다.')

        print('')
        print('+-'*20 + '+')
        print('Pretrained Checkpoint 생성')
        print('+-'*20 + '+')
        print('')

        config_path = tf_models_path+'/models/research/object_detection/configs/tf2/'

        download_config = config_path+base_pipeline_file
        download_tar = pretrained_checkpoint

        os.system(f"""
        wget {download_tar}
        """)

        for f in os.listdir(training_path):
            if f.endswith('.tar.gz'):
                fs = os.path.join(training_path, f)
                fsn = os.path.basename(fs).split('.')[0]
                os.system(f"""
                tar -xzvf {fs}
                rm -rf {fs}
                cp {download_config} ./
                """)
                if fsn != model_name:
                    os.system(f"""
                    mv {fsn} {model_name}
                    """)

        print('')
        print('+-'*20 + '+')
        print('Config File 생성')
        print('+-'*20 + '+')
        print('')

        pipeline_fname = training_path+'/'+base_pipeline_file
        fine_tune_checkpoint =  training_path+'/'+model_name+'/checkpoint/ckpt-0'
        num_classes = len(unique_classes)
        print('Total classes:', num_classes)

        train_record_fname = result_path+'/images/train.tfrecord'
        test_record_fname = result_path+'/images/test.tfrecord'
        label_map_pbtxt_fname = result_path+'/images/labelmap.pbtxt'

        def fix_batch_size_in_eval_config(config_text):
            pattern = r"(eval_config\s*{[^}]*?batch_size:\s*\d+);"
            fixed_text = re.sub(pattern, r"\1", config_text, flags=re.DOTALL)
            return fixed_text

        print('tuning:', tuning)

        pattern = r"\.?\d+(?:\.\d+)?(?:e-?\d+)?"

        if tuning.lower() == 'false':
            with open(pipeline_fname) as f:
                s = f.read()
            with open(training_path+'/pipeline.config', 'w') as f:
                s = re.sub('fine_tune_checkpoint: ".*?"', 'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
                #s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
                s = re.sub(r'input_path:\s*".*?train[^"]*"', f'input_path: "{train_record_fname}"', s)
                #s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)
                s = re.sub(r'input_path:\s*".*?val[^"]*"', f'input_path: "{test_record_fname}"', s)
                s = re.sub('label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)
                s = re.sub('batch_size: [0-9]+', 'batch_size: {}'.format(batch_size), s)
                s = re.sub('num_steps: [0-9]+', 'num_steps: {}'.format(num_steps), s)
                s = re.sub('num_classes: [0-9]+', 'num_classes: {}'.format(num_classes), s)
                s = re.sub('fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
                f.write(s)
                os.system(f"""
                rm -rf {pipeline_fname}
                """)
            with open('pipeline.config', 'r') as f:
                config_text = f.read()

            updated_config = fix_batch_size_in_eval_config(config_text)

            with open('pipeline.config', 'w') as f:
                f.write(updated_config)

        else:
            tuningOption_dict = json.loads(advancedSettingForObjectDetection)
            print(tuningOption_dict)
            print('imgSize')
            print(tuningOption_dict['imgSize'])
            print(type(tuningOption_dict['imgSize']))
            print('multiScale')
            print(tuningOption_dict['multiScale'])
            print(type(tuningOption_dict['multiScale']))

            with open(pipeline_fname) as f:
                s = f.read()
            with open(training_path+'/pipeline.config', 'w') as f:
                s = re.sub('fine_tune_checkpoint: ".*?"', 'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
                #s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
                s = re.sub(r'input_path:\s*".*?train[^"]*"', f'input_path: "{train_record_fname}"', s)
                #s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)
                s = re.sub(r'input_path:\s*".*?val[^"]*"', f'input_path: "{test_record_fname}"', s)
                s = re.sub('label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)
                s = re.sub('batch_size: [0-9]+', 'batch_size: {}'.format(batch_size), s)
                s = re.sub('num_steps: [0-9]+', 'num_steps: {}'.format(num_steps), s)
                s = re.sub('num_classes: [0-9]+', 'num_classes: {}'.format(num_classes), s)
                s = re.sub('fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
                f.write(s)
                os.system(f"""
                rm -rf {pipeline_fname}
                """)
            with open('pipeline.config', 'r') as f:
                config_text = f.read()

            updated_config = fix_batch_size_in_eval_config(config_text)

            with open('pipeline.config', 'w') as f:
                f.write(updated_config)


        print('')
        print('+-'*20 + '+')
        print('모델 훈련')
        print('+-'*20 + '+')
        print('')

        def db_mysql_training_update(projectId, versionId, trainProgress, epoch):
            cursor = db.cursor()
            try:
                sql = 'Update Training set trainProgress=%s, epoch=%s where (projectId, versionId)=%s'
                val = [trainProgress, epoch, (projectId, versionId)]
                cursor.execute(sql, val)
                db.commit()
            finally:
                cursor.close()        

        command = [
            "python", f"{tf_models_path}/models/research/object_detection/model_main_tf2.py",
            f"--pipeline_config_path={training_path}/pipeline.config",
            f"--model_dir={training_path}",
            f"--alsologtostderr",
            f"--sample_1_of_n_eval_examples=1",
            f"--checkpoint_every_n=100",
            f"--num_eval_steps=100"
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        classification_losses = []

        last_epoch = None
        last_progress = None
        max_epoch = None

        for line in process.stdout:
            print(line, end="", flush=True)

            match_epoch = re.search(r'Epoch (\d+) progress ([\d.]+)%', line)
            match_loss = re.search(r"'Loss/classification_loss': ([\d.]+)", line)

            if match_epoch:
                epoch = int(match_epoch.group(1))
                progress_value = float(match_epoch.group(2))

                if max_epoch is None or epoch > max_epoch:
                    max_epoch = epoch

                if epoch == max_epoch and progress_value == 100.0:
                    adjust_percent = 100.0 / max_epoch if max_epoch > 0 else 5.0
                    progress_value = 100.0 - adjust_percent                    
                    #progress_value = progress_value - 5.0

                if (epoch, progress_value) != (last_epoch, last_progress):
                    print(f"Epoch: {epoch}, Progress: {progress_value}%")
                    last_epoch, last_progress = epoch, progress_value

                db_mysql_training_update(
                    projectId = projectId,
                    versionId = versionId,
                    trainProgress = progress_value,
                    epoch = epoch
                )

            if match_loss:
                print(f"Captured loss: {match_loss.group(1)}")
                classification_loss = float(match_loss.group(1))
                classification_losses.append((epoch, classification_loss))

        process.wait()

        epochs = [epoch for epoch, loss in classification_losses]
        loss_values = [loss for epoch, loss in classification_losses]

        if classification_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, loss_values, marker='o', color='b', label='train')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{training_path}/metrics_chart.jpg")
            plt.close()
            metrics_chart = [file for file in os.listdir(training_path) if file.endswith('metrics_chart.jpg')]
            for i in metrics_chart:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=training_path+'/'+i)

            print("Metrics plot saved as 'metrics_chart.jpg'.")
        else:
            print("No classification loss values captured.")
        

        print('')
        print('+-'*20 + '+')
        print('모델 검증')
        print('+-'*20 + '+')        
        print('')

        validation_path = result_path+'/train/'

        command = [
            "python", f"{tf_models_path}/models/research/object_detection/model_main_tf2.py",
            f"--pipeline_config_path={training_path}/pipeline.config",
            f"--model_dir={training_path}",
            f"--checkpoint_dir={training_path}"
        ]

        eval_dir = f"{training_path}/eval"
        os.makedirs(eval_dir, exist_ok=True)

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        with open(f"{eval_dir}/evaluation.txt", "a") as f:
            for line in process.stdout:
                print(line, end="", flush=True)
                
                f.write(line)
                f.flush()

        process.wait()

        print('')
        print('+-'*20 + '+')
        print('mAP, Precision, Recall 확인')
        print('+-'*20 + '+')        
        print('')

        file_path = training_path+'/eval/evaluation.txt'
        with open(file_path, 'r') as f:
            lines = f.readlines()

        map_values = []
        precision_values = []
        recall_values = []

        for line in lines:
            if 'mAP' in line:
                map_values.append(float(line.split()[-1]))
            elif 'Precision' in line:
                precision_values.append(float(line.split()[-1]))
            elif 'Recall' in line:
                recall_values.append(float(line.split()[-1]))

        max_map = max(map_values)
        max_precision = max(precision_values)
        max_recall = max(recall_values)

        data = {
            'Metric': ['precision', 'recall', 'mAP'],
            'Value': [max_precision, max_recall, max_map]
        }

        df = pd.DataFrame(data)
        df.to_csv(result_path+'/metrics_score.csv')
        metrics_score = [file for file in os.listdir(result_path) if file.endswith('metrics_score.csv')]
        for i in metrics_score:
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)

        print(df.head())

        print('')
        print('+-'*20 + '+')
        print('모델 저장')
        print('+-'*20 + '+')
        print('')
            
        os.system(
        f"""
        python {tf_models_path}/models/research/object_detection/exporter_main_v2.py \
        --pipeline_config_path={training_path}/pipeline.config \
        --trained_checkpoint_dir={training_path} \
        --output_directory={training_path}/save
        """)

        training_model_save_path = training_path+'/save/saved_model'

        for root, dirs, files in os.walk(training_model_save_path):
            for file in files:
                local_file_path = os.path.join(root, file)

                relative_path = os.path.relpath(local_file_path, training_model_save_path)
                minio_object_path = f"{projectId}/{versionId}/train/model/train/weight/{relative_path}"

                client.fput_object(bucket_name=bucket, object_name=minio_object_path, file_path=local_file_path)
                print(f"Uploaded: {minio_object_path}")

        ### 모델 다운로드

        def install_p7zip():
            try:
                subprocess.run(['apt-get', 'install', '-y', 'p7zip-full'], check=True)
                print("p7zip-full이 성공적으로 설치되었습니다.")
            except subprocess.CalledProcessError as e:
                print(f"설치 중 오류가 발생했습니다: {e}")

        #install_p7zip()

        #def create_split_zip(input_path, output_path, zip_name="model_weight", split_size="1024M"):
        #    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        #    zip_file_base = os.path.join(output_path, zip_name)
        #    command = ["7z", "a", f"{zip_file_base}.7z", *files, f"-v{split_size}"]
        #    subprocess.run(command, check=True)
        #    print(f"분할 압축이 완료되었습니다: {zip_file_base}.7z")

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
                print(f"압축이 완료되었습니다: {zip_file_base}.7z")
            else:
                command = ["7z", "a", f"{zip_file_base}.7z", *files, f"-v{split_size}"]
                subprocess.run(command, check=True)
                print(f"분할 압축이 완료되었습니다: {zip_file_base}.7z")

        input_path = training_model_save_path
        output_path = training_model_save_path+'/weights_zip'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        create_zip(input_path, output_path, zip_name="model_weight", max_size=1024, split_size="1024M")

        zip_files = [file for file in os.listdir(output_path) if file.startswith('model_weight')]
        for i in zip_files:
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/model_weight/'+i, file_path=output_path+'/'+i)

        ########################################################################

        print('')
        print('+-'*20 + '+')
        print('테스트 데이터 추론 결과 저장')
        print('+-'*20 + '+')
        print('')        

        predict_dir = os.path.join(training_path, 'predict')
        os.makedirs(predict_dir, exist_ok=True)
        PATH_TO_SAVED_MODEL = training_path + '/save/saved_model'
        PATH_TO_INFERENCE_IMAGE = result_path + '/images/test'

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

        image_files = [f for f in os.listdir(PATH_TO_INFERENCE_IMAGE) if f.endswith('.jpg') or f.endswith('.png')]
        if not image_files:
            raise FileNotFoundError("No image files found for inference.")

        print(f"Found {len(image_files)} images for prediction.")

        for image_file in image_files:
            try:
                image_path = os.path.join(PATH_TO_INFERENCE_IMAGE, image_file)
                print(f"Processing: {image_path}")
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]
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

                result_image_path = os.path.join(predict_dir, image_file)
                result_image = Image.fromarray(image_np)
                result_image.save(result_image_path)
                print(f"Saved: {result_image_path}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        predicted_images = [f for f in os.listdir(predict_dir) if f.endswith('.jpg') or f.endswith('.png')]
        num_images = len(predicted_images)
        print(f"Total predicted images: {num_images}")        

        if num_images == 0:
            raise RuntimeError("No predicted images to visualize.")

        predicted_images = random.sample(predicted_images, min(num_images, 10))
        cols, rows = 3, 4
        image_count = len(predicted_images)
        images = [Image.open(os.path.join(predict_dir, img)) for img in predicted_images]
        width, height = images[0].size
        spacing = 10
        grid_width = cols * width + (cols - 1) * spacing
        grid_height = rows * height + (rows - 1) * spacing
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

        for i, img in enumerate(images):
            x_offset = (i % cols) * (width + spacing)
            y_offset = (i // cols) * (height + spacing)
            grid_image.paste(img, (x_offset, y_offset))

        grid_image_path = os.path.join(predict_dir, 'validation_result.jpg')
        grid_image.save(grid_image_path)
        validation_results = [file for file in os.listdir(predict_dir) if file.endswith('validation_result.jpg')]
        for i in validation_results:
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=predict_dir+'/'+i)        

        db_mysql_training_update(
            projectId = projectId,
            versionId = versionId,
            trainProgress = 100,
            epoch = epoch
        )

        from io import BytesIO
        def upload_text_to_minio(bucket_name, object_path, text):
            data = text.encode('utf-8')
            data_stream = BytesIO(data)
            client.put_object(
                bucket_name=bucket_name,
                object_name=object_path,
                data=data_stream,
                length=len(data),
                content_type='text/plain'
            )

        algorithm_path = f"{projectId}/{versionId}/train/algorithm.txt"
        #bucket_path = f"{projectId}/{versionId}/train/bucket.txt"
        upload_text_to_minio(bucket, algorithm_path, algorithm)
        #upload_text_to_minio(bucket, bucket_path, bucket)

        print(f"Validation result saved at: {grid_image_path}")        

    ################################################################################################
    ## preprocessing task 1 run
    ################################################################################################

    def db_mysql_stat_update(projectId, versionId, statusOfTrain):
        cursor = db.cursor()
        try:
            sql = 'Update Stat set statusOfTrain=%s where (projectId, versionId)=%s'
            val = [statusOfTrain, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()

    def db_mysql_training_update(projectId, versionId, algorithm, batchsize, mAP, recall, precisions, subStatusOfTraining):
        cursor = db.cursor()
        try:
            sql = 'Update Training set algorithm=%s, batchsize=%s, mAP=%s, recall=%s, precisions=%s, subStatusOfTraining=%s where (projectId, versionId)=%s'
            val = [algorithm, batchsize, mAP, recall, precisions, subStatusOfTraining, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()

    def db_mysql_training_tmp_update(projectId, versionId, subStatusOfTraining):
        cursor = db.cursor()
        try:
            sql = 'Update Training set subStatusOfTraining=%s where (projectId, versionId)=%s'
            val = [subStatusOfTraining, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()

    def db_mysql_training_error_update(projectId, versionId, TrainingErrorCategory, TrainingErrorLog):
        cursor = db.cursor()
        try:
            sql = 'Update Error set TrainingErrorCategory=%s, TrainingErrorLog=%s where (projectId, versionId)=%s'
            val = [TrainingErrorCategory, TrainingErrorLog, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()
        
    db = pymysql.connect(host=PyMySQLHost, user=PyMySQLUser, password=PyMySQLPassword, port=PyMySQLPort, db=PyMySQLDB, charset='utf8')

    try:
        if algorithm in ('efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2', 'efficientdet_d3', 'efficientdet_d4', 'efficientdet_d5', 'efficientdet_d6', 'efficientdet_d7'):
            db_mysql_stat_update(projectId = projectId, versionId = versionId, statusOfTrain = 'RUNNING')
            db_mysql_training_tmp_update(projectId = projectId, versionId = versionId, subStatusOfTraining = 'RUNNING')
            db_mysql_training_error_update(projectId = projectId, versionId = versionId, TrainingErrorCategory = 0, TrainingErrorLog = '정상')
            efficientdet(epoch)

    except Exception as e:
        db_mysql_stat_update(projectId = projectId, versionId = versionId, statusOfTrain = 'ERROR')
        db_mysql_training_tmp_update(projectId = projectId, versionId = versionId, subStatusOfTraining = 'ERROR')

        error_message = f"Error: {e}\n"

        with open(result_path+"/error.log", "w") as f:
            f.write(error_message)

        #error_log = [file for file in os.listdir(result_path) if file.endswith('error.log')]
        #for i in error_log:
        #    client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)

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
            else:
                trainingerrorcategory = 4

            #with open(result_path+"/error_message.txt", "w") as f:
            #    f.write(new_content)

        db_mysql_training_error_update(projectId = projectId, versionId = versionId, TrainingErrorCategory = trainingerrorcategory, TrainingErrorLog = error_message)

        print(f"Error: {e}")

        sys.exit(1)

    else:
        try:    
            metrics_file = f"{projectId}/{versionId}/train/metrics_score.csv"
            for item in client.list_objects(bucket_name=bucket, prefix=f"{projectId}/{versionId}/train", recursive=True):        
                if item.object_name == metrics_file:            
                    metrics = client.get_object(bucket, item.object_name)            
                    df_metrics = pd.read_csv(metrics, index_col=0, names=['metrics', 'value'], header=0)            
                    break    
            else:        
                print("Metrics file not found")        
                return        
            
            precision = df_metrics['value'][0]    
            recall = df_metrics['value'][1]    
            mAP = df_metrics['value'][2]

            db_mysql_training_update(projectId = projectId, versionId = versionId, algorithm = algorithm, batchsize = batchsize, mAP = mAP, recall = recall, precisions = precision, subStatusOfTraining = 'FINISH')
        except Exception as e:        
            print(f"Error while processing metrics: {e}")        
            sys.exit(1)

################################################################################################
## kubeflow pipeline upload
################################################################################################

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
