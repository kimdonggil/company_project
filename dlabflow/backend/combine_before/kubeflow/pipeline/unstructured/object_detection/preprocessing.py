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
from pydantic import BaseModel, ValidationError 
import pydantic
import typing as t
from typing import Any, Type
import requests
import json
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
KubeflowPieplineName = os.getenv('KubeflowPieplinePreprocessing')
KubeflowGPUName = os.getenv('KubeflowGPUName1')
KubeflowGPUValue = os.getenv('KubeflowGPUValue1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:unstructured-objectdetection-20251031-beta')
def Preprocessing(projectId: str, versionId: str, dataPath: str, dataNormalization: str, dataAugmentation: str, trainRatio: int, validationRatio: int, testRatio: int):
    import os
    import glob
    import xml.etree.ElementTree as ET
    import cv2
    import random
    import matplotlib.pyplot as plt
    import pandas as pd
    from distutils.dir_util import copy_tree
    import splitfolders
    import shutil
    from minio import Minio
    import csv
    from pathlib import Path
    import pymysql
    import sys
    from collections import defaultdict
    from sklearn.model_selection import train_test_split
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
    base_path = '/mnt/dlabflow/backend/minio/'+bucket
    tmp_annotation_path = base_path+'/'+projectId+'/rawdata/annotations/'
    tmp_image_path = base_path+'/'+projectId+'/rawdata/images/'
    minio_path = '/mnt/dlabflow/backend/minio/'+bucket
    annotation_path = tmp_annotation_path
    annotation_list = sorted([f for f in os.listdir(annotation_path) if f.endswith(".xml")])
    result_path = minio_path+'/'+projectId+'/'+versionId+'/preprocessing'
    tmps = result_path+'/tmp/a'
    tmps_split = result_path+'/tmpsplit'
    client = Minio(endpoint=MinioEndpoint, access_key=MinioAccessKey, secret_key=MinioSecretKey, secure=MinioSecure)
    db = pymysql.connect(host=PyMySQLHost, user=PyMySQLUser, password=PyMySQLPassword, port=PyMySQLPort, db=PyMySQLDB, charset='utf8')

    def db_mysql_stat_update(projectId, versionId, statusOfPreprocessing):
        cursor = db.cursor()
        try:
            sql = 'Update Stat set statusOfPreprocessing=%s where (projectId, versionId)=%s'
            val = [statusOfPreprocessing, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()

    def db_mysql_preprocessing_update(projectId, versionId, numOfTrain, numOfTest, numOfValidation, numOfRaw, numOfAugmentation, numOfAugmentationRaw):
        cursor = db.cursor()
        try:
            sql = 'Update Preprocessing set numOfTrain=%s, numOfTest=%s, numOfValidation=%s, numOfRaw=%s, numOfAugmentation=%s, numOfAugmentationRaw=%s where (projectId, versionId)=%s'
            val = [numOfTrain, numOfTest, numOfValidation, numOfRaw, numOfAugmentation, numOfAugmentationRaw, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()

    def preprocessings():
        def create_base_annotation(filename, width, height, depth=3):
            root = ET.Element('annotation')
            ET.SubElement(root, 'folder').text = ''
            ET.SubElement(root, 'filename').text = filename
            ET.SubElement(root, 'path').text = filename
            source = ET.SubElement(root, 'source')
            ET.SubElement(source, 'database').text = 'dlabflow'
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = str(depth)
            ET.SubElement(root, 'segmented').text = '0'
            return root

        def dataframe(annotation_path, annotation_list):
            df = []
            for ann in annotation_list:
                tree = ET.parse(os.path.join(annotation_path, ann))
                root = tree.getroot()
                filename = os.path.splitext(ann)[0]
                for obj in root.findall('object'):
                    b = obj.find('bndbox')
                    df.append([filename, obj.find('name').text, int(b.find('xmin').text), int(b.find('ymin').text), int(b.find('xmax').text), int(b.find('ymax').text)])
            df = pd.DataFrame(df, columns=['filename','label','xmin','ymin','xmax','ymax'])
            grouped = df.groupby('filename').apply(lambda x: list(zip(x.xmin, x.ymin, x.xmax, x.ymax, x.label))).to_dict()
            grouped = {k.lower(): v for k, v in grouped.items()}
            return grouped

        def original(normalization_type):
            out_dir = os.path.join(result_path, 'normalization', normalization_type)
            os.makedirs(out_dir, exist_ok=True)            
            image_files = {os.path.splitext(f)[0]: f for f in os.listdir(tmp_image_path) if os.path.splitext(f)[1].lower() in ('.jpg', '.png')}
            annotation_files = {os.path.splitext(f)[0]: f for f in os.listdir(tmp_annotation_path) if f.endswith(".xml")}
            for key, img_file in image_files.items():
                src_img = os.path.join(tmp_image_path, img_file)
                name, ext = os.path.splitext(img_file)
                new_img_name = f"{name}_{normalization_type}{ext}"
                dst_img = os.path.join(out_dir, new_img_name)
                shutil.copy2(src_img, dst_img)
                if key in annotation_files:
                    xml_file = annotation_files[key]
                    src_xml = os.path.join(tmp_annotation_path, xml_file)
                    tree = ET.parse(src_xml)
                    root = tree.getroot()
                else:
                    image = cv2.imread(src_img)
                    H, W = image.shape[:2]
                    root = create_base_annotation(new_img_name, W, H, depth=3)
                    ET.SubElement(root, 'folder').text = 'images'
                    ET.SubElement(root, 'filename').text = new_img_name
                    ET.SubElement(root, 'path').text = os.path.join('images', new_img_name)
                    size = ET.SubElement(root, "size")
                    ET.SubElement(size, 'width').text = str(W)
                    ET.SubElement(size, 'height').text = str(H)
                    ET.SubElement(size, 'depth').text = "3"
                    ET.SubElement(root, 'segmented').text = "0"
                    tree = ET.ElementTree(root)
                root.find('filename').text = new_img_name
                if root.find('path') is not None:
                    root.find('path').text = new_img_name
                tree.write(os.path.join(out_dir, f"{key}_{normalization_type}.xml"))

        def grayscale(normalization_type):
            out_dir = os.path.join(result_path, 'normalization', normalization_type)
            os.makedirs(out_dir, exist_ok=True)
            image_files = {os.path.splitext(f)[0]: f for f in os.listdir(tmp_image_path) if os.path.splitext(f)[1].lower() in ('.jpg', '.png')}
            annotation_files = {os.path.splitext(f)[0]: f for f in annotation_list}
            all_keys = image_files.keys()
            for key in all_keys:
                img_file = image_files[key]
                img_path = os.path.join(tmp_image_path, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                name, ext = os.path.splitext(img_file)
                new_filename = f"{name}_{normalization_type}{ext.lower()}"
                out_img_path = os.path.join(out_dir, new_filename)
                cv2.imwrite(out_img_path, image_gray)
                if key in annotation_files:
                    xml_file = annotation_files[key]
                    xml_path = os.path.join(annotation_path, xml_file)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                else:
                    H, W = image_gray.shape[:2]
                    root = create_base_annotation(new_filename, W, H, depth=3)
                    ET.SubElement(root, 'folder').text = 'images'
                    ET.SubElement(root, 'filename').text = new_filename
                    ET.SubElement(root, 'path').text = os.path.join('images', new_filename)
                    size = ET.SubElement(root, 'size')
                    ET.SubElement(size, 'width').text = str(W)
                    ET.SubElement(size, 'height').text = str(H)
                    ET.SubElement(size, 'depth').text = '1'
                    ET.SubElement(root, 'segmented').text = '0'
                    tree = ET.ElementTree(root)
                root.find('filename').text = new_filename
                if root.find('path') is not None:
                    root.find('path').text = new_filename
                tree.write(os.path.join(out_dir, f"{key}_{normalization_type}.xml"))

        def reverse(image_path):
            out_dir = os.path.join(result_path, 'augmentation', augmentation_type)
            os.makedirs(out_dir, exist_ok=True)
            image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_path) if os.path.splitext(f)[1].lower() in ('.jpg', '.png')}
            annotation_files = {os.path.splitext(f)[0]: f for f in os.listdir(annotation_path) if f.endswith('.xml')}
            grouped_bboxes = {}
            if annotation_files:
                grouped_bboxes = dataframe(annotation_path, list(annotation_files.values()))
            def flip_bbox(reverse_select, bbox, W, H):
                x_min, y_min, x_max, y_max = bbox
                if reverse_select == 0:
                    return (x_min, H - y_max, x_max, H - y_min)
                else:
                    return (W - x_max, y_min, W - x_min, y_max)
            for key, img_file in image_files.items():
                base_key = key.split('_')[0].lower()
                img_path = os.path.join(image_path, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                H, W = image.shape[:2]
                flipped = cv2.flip(image, reverse_select)
                new_img_name = f"{key}_{augmentation_type}.jpg"
                cv2.imwrite(os.path.join(out_dir, new_img_name), flipped)
                if base_key in annotation_files:
                    tree = ET.parse(os.path.join(annotation_path, annotation_files[base_key]))
                    root = tree.getroot()
                else:
                    root = create_base_annotation(new_img_name, W, H, depth=3)
                    tree = ET.ElementTree(root)
                    ET.SubElement(root, 'folder').text = 'images'
                filename_tag = root.find('filename')
                if filename_tag is None:
                    filename_tag = ET.SubElement(root, 'filename')
                filename_tag.text = new_img_name
                path_tag = root.find('path')
                if path_tag is None:
                    path_tag = ET.SubElement(root, 'path')
                path_tag.text = new_img_name
                for obj in root.findall('object'):
                    root.remove(obj)
                if base_key in grouped_bboxes:
                    for (xmin, ymin, xmax, ymax, label) in grouped_bboxes[base_key]:
                        fxmin, fymin, fxmax, fymax = flip_bbox(reverse_select, (xmin, ymin, xmax, ymax), W, H)
                        obj = ET.SubElement(root, 'object')
                        ET.SubElement(obj, 'name').text = label
                        bnd = ET.SubElement(obj, 'bndbox')
                        ET.SubElement(bnd, 'xmin').text = str(fxmin)
                        ET.SubElement(bnd, 'ymin').text = str(fymin)
                        ET.SubElement(bnd, 'xmax').text = str(fxmax)
                        ET.SubElement(bnd, 'ymax').text = str(fymax)
                tree.write(os.path.join(out_dir, f"{key}_{augmentation_type}.xml"))

        def rotation(image_path):
            out_dir = os.path.join(result_path, 'augmentation', augmentation_type)
            os.makedirs(out_dir, exist_ok=True)
            image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_path) if os.path.splitext(f)[1].lower() in ('.jpg', '.png')}
            annotation_files = {os.path.splitext(f)[0]: f for f in os.listdir(annotation_path) if f.endswith('.xml')}
            grouped_bboxes = {}
            if annotation_files:
                grouped_bboxes = dataframe(annotation_path, list(annotation_files.values()))
            def rotate_bbox(xmin, ymin, xmax, ymax, W, H, t):
                if t == 'rotation_90':
                    return (H - ymax, xmin, H - ymin, xmax)
                elif t == 'rotation_180':
                    return (W - xmax, H - ymax, W - xmin, H - ymin)
                elif t == 'rotation_270':
                    return (ymin, W - xmax, ymax, W - xmin)
            rotate_map = {'rotation_90': cv2.ROTATE_90_CLOCKWISE, 'rotation_180': cv2.ROTATE_180, 'rotation_270': cv2.ROTATE_90_COUNTERCLOCKWISE}
            for key, img_file in image_files.items():
                base_key = key.split('_')[0].lower()
                img_path = os.path.join(image_path, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                H, W = image.shape[:2]
                rotated = cv2.rotate(image, rotate_map[augmentation_type])
                new_img_name = f"{key}_{augmentation_type}.jpg"
                cv2.imwrite(os.path.join(out_dir, new_img_name), rotated)
                if base_key in annotation_files:
                    tree = ET.parse(os.path.join(annotation_path, annotation_files[base_key]))
                    root = tree.getroot()
                else:
                    root = create_base_annotation(new_img_name, W, H, depth=3)
                    tree = ET.ElementTree(root)
                    ET.SubElement(root, 'folder').text = 'images'
                filename_tag = root.find('filename')
                if filename_tag is None:
                    filename_tag = ET.SubElement(root, 'filename')
                filename_tag.text = new_img_name
                path_tag = root.find('path')
                if path_tag is None:
                    path_tag = ET.SubElement(root, 'path')
                path_tag.text = new_img_name
                for obj in root.findall('object'):
                    root.remove(obj)
                if base_key in grouped_bboxes:
                    for (xmin, ymin, xmax, ymax, label) in grouped_bboxes[base_key]:
                        fxmin, fymin, fxmax, fymax = rotate_bbox(xmin, ymin, xmax, ymax, W, H, augmentation_type)
                        obj = ET.SubElement(root, 'object')
                        ET.SubElement(obj, 'name').text = label
                        bnd = ET.SubElement(obj, 'bndbox')
                        ET.SubElement(bnd, 'xmin').text = str(fxmin)
                        ET.SubElement(bnd, 'ymin').text = str(fymin)
                        ET.SubElement(bnd, 'xmax').text = str(fxmax)
                        ET.SubElement(bnd, 'ymax').text = str(fymax)
                tree.write(os.path.join(out_dir, f"{key}_{augmentation_type}.xml"))            

        def brightness(image_path, brightness=0):
            out_dir = os.path.join(result_path, 'augmentation', augmentation_type)
            os.makedirs(out_dir, exist_ok=True)
            image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_path) if os.path.splitext(f)[1].lower() in ('.jpg', '.png')}
            annotation_files = {os.path.splitext(f)[0]: f for f in os.listdir(annotation_path) if f.endswith(".xml")}
            for key, img_file in image_files.items():
                base_key = key.split('_')[0].lower()
                img_path = os.path.join(image_path, img_file)
                image = cv2.imread(img_path)
                image_brightness = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)
                name, ext = os.path.splitext(img_file)
                new_img_name = f"{key}_{augmentation_type}{ext.lower()}"
                out_img_path = os.path.join(out_dir, new_img_name)
                cv2.imwrite(out_img_path, image_brightness)
                if base_key in annotation_files:  
                    xml_file = annotation_files[base_key]
                    xml_path = os.path.join(annotation_path, xml_file)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                else:
                    H, W = image_brightness.shape[:2]
                    root = create_base_annotation(new_img_name, W, H, depth=3)
                    ET.SubElement(root, 'folder').text = 'images'
                    ET.SubElement(root, 'filename').text = new_img_name
                    ET.SubElement(root, 'path').text = new_img_name
                    size = ET.SubElement(root, 'size')
                    ET.SubElement(size, 'width').text = str(W)
                    ET.SubElement(size, 'height').text = str(H)
                    ET.SubElement(size, 'depth').text = str(image_brightness.shape[2])
                    ET.SubElement(root, 'segmented').text = '0'
                    tree = ET.ElementTree(root)
                root.find('filename').text = new_img_name
                if root.find('path') is not None:
                    root.find('path').text = new_img_name
                tree.write(os.path.join(out_dir, f"{key}_{augmentation_type}.xml"))

        def contrast(image_path, contrast=1.0):
            out_dir = os.path.join(result_path, 'augmentation', augmentation_type)
            os.makedirs(out_dir, exist_ok=True)
            image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_path) if os.path.splitext(f)[1].lower() in ('.jpg', '.png')}
            annotation_files = {os.path.splitext(f)[0]: f for f in os.listdir(annotation_path) if f.endswith(".xml")}
            for key, img_file in image_files.items():
                base_key = key.split('_')[0].lower()
                img_path = os.path.join(image_path, img_file)
                image = cv2.imread(img_path)
                image_contrast = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
                name, ext = os.path.splitext(img_file)
                new_img_name = f"{key}_{augmentation_type}{ext.lower()}"
                out_img_path = os.path.join(out_dir, new_img_name)
                cv2.imwrite(out_img_path, image_contrast)
                if base_key in annotation_files:
                    xml_file = annotation_files[base_key]
                    xml_path = os.path.join(annotation_path, xml_file)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                else:
                    H, W = image_contrast.shape[:2]
                    root = create_base_annotation(new_img_name, W, H, depth=3)
                    ET.SubElement(root, 'folder').text = 'images'
                    ET.SubElement(root, 'filename').text = new_img_name
                    ET.SubElement(root, 'path').text = new_img_name
                    size = ET.SubElement(root, 'size')
                    ET.SubElement(size, 'width').text = str(W)
                    ET.SubElement(size, 'height').text = str(H)
                    ET.SubElement(size, 'depth').text = str(image_contrast.shape[2])
                    ET.SubElement(root, 'segmented').text = '0'
                    tree = ET.ElementTree(root)
                root.find('filename').text = new_img_name
                if root.find('path') is not None:
                    root.find('path').text = new_img_name
                tree.write(os.path.join(out_dir, f"{key}_{augmentation_type}.xml"))

        rename_map = {'rotation_LR': 'reverse_lr', 'rotation_TB': 'reverse_tb'}
        check_dataAugmentation = [x.strip() for x in dataAugmentation.replace(',', ' ').split() if x.strip()]
        check_dataAugmentation = [rename_map.get(x, x) for x in check_dataAugmentation]
        Brightness = next((int(x.split('_')[1]) for x in check_dataAugmentation if x.startswith('brightness_')), 0)
        Brightness_percent = int(255 * (Brightness / 100))
        Contrast = next((float(x.split('_')[1]) for x in check_dataAugmentation if x.startswith('contrast_')), 0.0)
        normalization_type = 'grayscale' if 'grayscale' in dataNormalization else 'original'
        if normalization_type == 'grayscale':
            grayscale(normalization_type)
        else:
            original(normalization_type)
        normalized_dir = os.path.join(result_path, 'normalization', normalization_type)
        for aug in check_dataAugmentation:
            if aug == 'reverse_lr':
                augmentation_type = 'reverse_left_right'
                reverse_select = 1
                reverse(normalized_dir)
            elif aug == 'reverse_tb':
                augmentation_type = 'reverse_top_bottom'
                reverse_select = 0
                reverse(normalized_dir)
            elif aug in ['rotation_90', 'rotation_180', 'rotation_270']:
                augmentation_type = aug
                rotation(normalized_dir)
            elif aug.startswith('brightness_'):
                augmentation_type = 'brightness'
                brightness(normalized_dir, brightness=Brightness_percent)
            elif aug.startswith('contrast_'):
                augmentation_type = 'contrast'
                contrast(normalized_dir, contrast=Contrast)

    def split_dataset(base_dir, allow_ext=['.jpg', '.png', '.bmp'], train_ratio=0.7, val_ratio=0.15):
        try:
            output_dir = f'{base_dir}/datasplit'
            train_dir = os.path.join(output_dir, 'train')
            val_dir = os.path.join(output_dir, 'val')
            test_dir = os.path.join(output_dir, 'test')
            for d in [train_dir, val_dir, test_dir]:
                os.makedirs(d, exist_ok=True)

            images = sorted([
                str(p) for p in Path(base_dir).rglob('*')
                if p.suffix.lower() in allow_ext and not str(p).startswith(output_dir)
            ])
            print(f"총 이미지 수: {len(images)}")
            total_count = len(images)

            n_train = round(total_count * train_ratio)
            n_val = round(total_count * val_ratio)
            n_test = total_count - n_train - n_val

            train_files = images[:n_train]
            val_files = images[n_train:n_train + n_val]
            test_files = images[n_train + n_val:]

            def copy_pairs(file_list, target_dir):
                for img in file_list:
                    img_path = Path(img)
                    xml = img_path.with_suffix('.xml')
                    shutil.copy(img, target_dir)
                    if xml.exists():
                        shutil.copy(str(xml), target_dir)

            copy_pairs(train_files, train_dir)
            copy_pairs(val_files, val_dir)
            copy_pairs(test_files, test_dir)

            def count_files(folder, img_ext=allow_ext):
                img_count = len([p for p in Path(folder).glob('*') if p.suffix.lower() in img_ext])
                xml_count = len(list(Path(folder).glob('*.xml')))
                return img_count, xml_count

            train_count, _ = count_files(train_dir)
            val_count, _ = count_files(val_dir)
            test_count, _ = count_files(test_dir)
            raw_count, _ = count_files(base_dir)

            db_mysql_preprocessing_update(
                projectId=projectId,
                versionId=versionId,
                numOfTrain=train_count,
                numOfTest=test_count,
                numOfValidation=val_count,
                numOfRaw=raw_count,
                numOfAugmentation=train_count + val_count + test_count - raw_count,
                numOfAugmentationRaw=train_count + val_count + test_count
            )
            db_mysql_stat_update(projectId, versionId, 'FINISH')

            for name, folder in zip(['Train', 'Val', 'Test'], [train_dir, val_dir, test_dir]):
                imgs, xmls = count_files(folder)
                print(f"{name}: 이미지 {imgs}개, XML {xmls}개")

        except Exception:
            db_mysql_stat_update(projectId=projectId, versionId=versionId, statusOfPreprocessing='ERROR')
            print('데이터 분리 과정에서 오류가 발생하였습니다.')
            raise

    """ task """
    try:
        preprocessings()
        split_dataset(base_dir=result_path, train_ratio=trainRatio/100, val_ratio=validationRatio/100)
    except Exception:
        db_mysql_stat_update(projectId=projectId, versionId=versionId, statusOfPreprocessing='ERROR')
        raise

def list_type(v):
    return re.sub(r'[\[\]]', '', v)

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--dataPath', type=list_type)
    parser.add_argument('--dataNormalization', type=list_type)
    parser.add_argument('--dataAugmentation', type=list_type)
    parser.add_argument('--trainRatio', type=int)
    parser.add_argument('--validationRatio', type=int)
    parser.add_argument('--testRatio', type=int)
    args = parser.parse_args()
    Preprocessing_apply = Preprocessing(args.projectId, args.versionId, args.dataPath, args.dataNormalization, args.dataAugmentation, args.trainRatio, args.validationRatio, args.testRatio) \
        .set_display_name('Data Preprocessing') \
        .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}"))

    Preprocessing_apply.execution_options.caching_strategy.max_cache_staleness = 'P0D'

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
