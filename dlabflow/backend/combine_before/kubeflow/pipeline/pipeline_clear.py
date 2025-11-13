from kubernetes import client, config
from datetime import datetime, timezone, timedelta
import os
import requests
import kfp
from dotenv import load_dotenv

dotenv_path = '/mnt/project/dlabflow/backend/config'
load_dotenv(dotenv_path)

KubeflowPipelineAutoannotation = os.getenv('KubeflowPipelineAutoannotation')
KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')

def delete_old_pods(namespace: str, max_seconds: int = 3600):
    try:
        config.load_kube_config()
    except:
        config.load_incluster_config()

    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=namespace).items
    now = datetime.now(timezone.utc)

    deleted_any = False

    for pod in pods:
        if not pod.status.container_statuses:
            continue

        terminated_containers = [cs.state.terminated for cs in pod.status.container_statuses if cs.state.terminated]
        if not terminated_containers:
            continue

        created_at = pod.metadata.creation_timestamp
        min_seconds = (now - created_at).total_seconds()
        if min_seconds < max_seconds:
            continue

        pod_name = pod.metadata.name
        print(f"Deleting pod: {pod_name}")
        try:
            v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
            deleted_any = True
        except client.exceptions.ApiException as e:
            print(f"Failed to delete pod {pod_name}: {e}")

    if not deleted_any:
        print("No pods to delete")

def delete_recent_kfp_runs(experiment_name: str, namespace: str, max_seconds: int = 3600):
    HOST = f"http://{KubeflowHost}"
    session = requests.Session()
    response = session.get(HOST)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'login': KubeflowUsername, 'password': KubeflowPassword}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()['authservice_session']

    client = kfp.Client(
        host=f"{HOST}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=namespace
    )

    experiment = client.get_experiment(experiment_name=experiment_name)
    runs_obj = client.list_runs(experiment_id=experiment.id)

    if not runs_obj or not runs_obj.runs:
        print("No pipeline runs to delete")
        return    

    runs = runs_obj.runs
    cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=max_seconds)

    deleted_any = False
    for r in runs:
        run_created = r.created_at
        if run_created.tzinfo is None:
            run_created = run_created.replace(tzinfo=timezone.utc)
        if run_created < cutoff_time:
            print(f"Deleting run: {r.name} ({r.id})")
            client._run_api.delete_run(r.id)
            deleted_any = True

    if not deleted_any:
        print("No pipeline to delete")

if __name__ == "__main__":
    max_seconds = 3600
    print("Step 1: Delete old Kubernetes pods")
    delete_old_pods(KubeflowNamespace, max_seconds=max_seconds)

    print("\nStep 2: Delete recent Kubeflow pipeline runs")
    delete_recent_kfp_runs(KubeflowPipelineAutoannotation, KubeflowNamespace, max_seconds=max_seconds)

