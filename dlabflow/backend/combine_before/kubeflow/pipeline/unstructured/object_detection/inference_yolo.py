from functools import partial
from kfp.components import create_component_from_func
from kfp import compiler, dsl, onprem
from kubernetes.client import V1EnvVar
import argparse
import requests
import kfp
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
KubeflowPieplineName = os.getenv('KubeflowPieplineInference')
KubeflowGPUName = os.getenv('KubeflowGPUName4')
KubeflowGPUValue = os.getenv('KubeflowGPUValue4')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:unstructured-objectdetection-20251031-beta')
def Inference(projectId: str, versionId: str, sessionId: str):
    import os
    from datetime import datetime
    from pathlib import Path
    from minio import Minio
    from ultralytics import YOLO
    import pymysql
    import torch
    import shutil
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
    os.makedirs(inference_before_path, exist_ok=True)
    os.makedirs(inference_after_path, exist_ok=True)
    client = Minio(endpoint=MinioEndpoint, access_key=MinioAccessKey, secret_key=MinioSecretKey, secure=MinioSecure)

    for item in client.list_objects(bucket, prefix=f'{projectId}/{versionId}/inference/{sessionId}/before', recursive=True):
        client.fget_object(bucket, item.object_name, f'{minio_path}/{item.object_name}')

    def predict():
        if torch.cuda.is_available():
            model_dir = f'{minio_path}/{projectId}/{versionId}/inference/{sessionId}/algorithm'
            os.makedirs(model_dir, exist_ok=True)
            for item in client.list_objects(bucket, prefix=f'{projectId}/{versionId}/train/model/train/weight', recursive=True):
                client.fget_object(bucket, item.object_name, f'{model_dir}/{Path(item.object_name).name}')
            model = YOLO(f'{model_dir}/best.pt')
            results = model.predict(source=inference_before_path, save=False)
            save_dir = f"{minio_path}/{projectId}/{versionId}/inference/{sessionId}/after"
            os.makedirs(save_dir, exist_ok=True)
            for i, result in enumerate(results):
                boxes = result.boxes
                #if len(boxes) == 0:
                #    continue
                confs = boxes.conf.cpu()
                classes = boxes.cls.cpu()
                xyxy = boxes.xyxy.cpu()
                """
                best_indices = []
                for cls_id in torch.unique(classes):
                    cls_mask = classes == cls_id
                    cls_confs = confs[cls_mask]
                    max_idx = torch.argmax(cls_confs)
                    best_indices.append(torch.arange(len(classes))[cls_mask][max_idx].item())
                result.boxes = result.boxes[best_indices]
                """
                save_path = os.path.join(save_dir, f"result_{i}.jpg")
                result.save(filename=save_path)
        else:
            print('[ERROR] GPU is not available.')

    def update_inference_status(status: str):
        conn = pymysql.connect(host=PyMySQLHost, user=PyMySQLUser, password=PyMySQLPassword, port=PyMySQLPort, db=PyMySQLDB, charset='utf8')
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE Inference SET statusOfInference=%s WHERE projectId=%s AND versionId=%s"
                cursor.execute(sql, (status, projectId, versionId))
            conn.commit()
        finally:
            conn.close()

    """ task """
    try:
        update_inference_status('RUNNING')
        prefix = f'{projectId}/{versionId}/inference/{sessionId}/after'
        objects_to_delete = client.list_objects(bucket, prefix=prefix, recursive=True)
        for obj in objects_to_delete:
            try:
                client.remove_object(bucket, obj.object_name)
                print(f'Deleted {obj.object_name}')
            except Exception as e:
                print(f'Failed to delete {obj.object_name}: {e}')
        predict()
        for item in os.listdir(inference_after_path):
            client.fput_object(bucket, f'{projectId}/{versionId}/inference/{sessionId}/after/{item}', f'{inference_after_path}/{item}')
        cleanup_keys = [
            f'{projectId}_{versionId}.csv',
            f'{projectId}/{versionId}/inference/{sessionId}/before/.keep',
            f'{projectId}/{versionId}/inference/{sessionId}/after/.keep'
        ]
        for key in cleanup_keys:
            client.remove_object(bucket, key)            
        update_inference_status('FINISH')
    except Exception as e:
        print(f'Inference failed: {e}')
        update_inference_status('ERROR')
        raise

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
