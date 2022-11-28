FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    DEBIAN_FRONTEND=noninteractive 

ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENVIRONMENT_PATH/bin:$PATH"

# Install Python3.8, poetry and redis-server
RUN apt-get update && \
    apt remove python-pip python3-pip && \
    apt-get install --no-install-recommends -y \
    redis-server \
    ca-certificates \
    g++ \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    cd /tmp && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3.8 1 && \
    curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock
RUN poetry install --only main

COPY ./src/ /app/

COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ./start.sh