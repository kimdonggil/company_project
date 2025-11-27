# GPU Resource Monitoring

`app 폴더`의 **main.py 파일**은 GPU 자원 정보를 수집하는 역할을 하며, **start.sh 스크립트를 통해 실행**할 수 있습니다.

- GPU API 요청에 사용하는 **포트(8123)는 필요에 따라 변경** 가능합니다.

`dashboard 폴더`는 **GPU 자원 모니터링 대시보드**를 포함하며, **start.sh 스크립트를 통해 실행**할 수 있습니다.

- 대시보드는 GPU 상태, 사용량, 노드 정보를 시각적으로 확인할 수 있습니다.

그 외 `gpu_test.py` 파일은 GPU 자원 모니터링 상태를 테스트하는데 사용되며, 실제 대시보드 실행 전에 GPU 정보 수집이 정상적으로 동작하는지 확인할 수 있습니다.

![Static Badge](https://img.shields.io/badge/fastapi-0.110.3-green)
![Static Badge](https://img.shields.io/badge/reflex-0.5.10.post1-green)

### 실행 결과

![Image](https://github.com/user-attachments/assets/a801214f-c0f9-44d4-94d4-17e17cc0aeff)
