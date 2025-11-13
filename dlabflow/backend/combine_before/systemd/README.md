# D-Lab Flow

`systemd` 폴더에서 ***.service 파일은 etc/systemd/system` 경로에 위치**해야 합니다.

- systemd에서 해당 파일을 새로 만들거나 수정한 후에는 반드시 daemon-reload 명령어를 실행하여 systemd 데몬에 변경 사항을 반영해야 합니다.
- 이후, enable 명령어로 kubeflow.service를 부팅 시 자동으로 실행되도록 활성화할 수 이므ㅕ, status 명령어를 사용해 kubeflow.service의 현재 상태를 확인할 수 있습니다.

kubeflow.service 실행 예시:

```
sudo systemctl daemon-reload
sudo systemctl enable kubeflow.service
sudo systemctl stop kubeflow.service
sudo systemctl restart kubeflow.service
sudo systemctl status kubeflow.service
```

```
● kubeflow.service - Kubeflow Service
     Loaded: loaded (/etc/systemd/system/kubeflow.service; enabled; vendor preset: enabled)
     Active: active (running) since Mon 2025-07-14 09:05:25 KST; 3 days ago
   Main PID: 2530051 (bash)
      Tasks: 49 (limit: 629145)
     Memory: 32.6M
     CGroup: /system.slice/kubeflow.service
             ├─2530051 /bin/bash /mnt/dlabflow/backend/kubeflow/start.sh
             └─2530052 kubectl

 7월 17 17:41:12 h100 bash[2530052]: Handling connection for 8080
```