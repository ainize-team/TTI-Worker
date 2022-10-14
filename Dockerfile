FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt remove python-pip  python3-pip \
    && apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3.8 1

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY ./src/ /app/

COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ./start.sh