FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        python3-pip \
        python3-dev \
        git \
        wget \
        tar \
        tzdata \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libnss3 \
        libatk-bridge2.0-0 \
        libcups2 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libxkbcommon0 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        curl && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo "$TZ" > /etc/timezone && \
    apt-get purge -y tzdata && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.tar.gz && \
    tar -zxvf cmake-3.28.1-linux-x86_64.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake-3.28.1-linux-x86_64.tar.gz

RUN pip install --upgrade pip

RUN git clone --recursive https://github.com/microsoft/LightGBM.git && \
    cd LightGBM && \
    mkdir build && cd build && \
    cmake -DUSE_CUDA=ON -DUSE_GPU=1 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DBUILD_PYTHON=ON .. && \
    cmake --build . -j$(nproc) && \
    cd .. && \
    sh build-python.sh install --precompile && \
    cd .. && rm -rf LightGBM

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install pycaret[full]

RUN pip uninstall -y plotly kaleido && \
    pip install plotly==5.24.1 kaleido==0.2.1

WORKDIR /app

