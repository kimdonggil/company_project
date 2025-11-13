# D-Lab Flow

`pipeline` 폴더는 **structured, unstructured 폴더로 구성**되어 있으며, pipeline_clear.py 파일을 통해 1시간 이후 kubernetes에서 완료(Completed), 실패(Error) Pod를 삭제하고, kubeflow에서 해당 Pod와 동일한 상태의 리소스를 삭제할 수 있습니다.