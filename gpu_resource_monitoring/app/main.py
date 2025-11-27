from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pynvml import (
    nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlShutdown
)

class GPUInfo(BaseModel):
    vendor: str
    index: int
    name: str
    mem_used: int
    mem_total: int
    mem_used_percent: float

app = FastAPI(title="GPU Monitor API", version="1.0.0")

def get_nvidia_gpu_info() -> List[GPUInfo]:
    gpus = []
    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        for i in range(count):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GPUInfo(
                    vendor="NVIDIA",
                    index=i,
                    name=f"NVIDIA GPU {i}",
                    mem_used=info.used // (1024**2),
                    mem_total=info.total // (1024**2),
                    mem_used_percent=round(info.used / info.total * 100, 2)
                )
            )
    except Exception:
        pass
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return gpus

@app.get("/gpu/", response_model=List[GPUInfo])
def read_gpus():
    return get_nvidia_gpu_info()

@app.get("/")
def root():
    return {"message": "GPU Monitor API"}
