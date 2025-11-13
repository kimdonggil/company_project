kubectl -n kubeflow-user-example-com apply -f pv.yaml
kubectl -n kubeflow-user-example-com apply -f pvc.yaml
kubectl -n kubeflow-user-example-com apply -f pod.yaml
