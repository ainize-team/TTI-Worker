FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# set environment variables
ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    VIRTUAL_ENVIRONMENT_PATH="/app/.venv" \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt remove python-pip  python3-pip \
    && apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.9 get-pip.py 

ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENVIRONMENT_PATH/bin:$PATH"

# Install Poetry
# https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    git \
    openssh-client \
    && curl -sSL https://install.python-poetry.org | python3.9 - 

WORKDIR /app
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock
RUN poetry install --only main

COPY ./src/ /app/

COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ./start.sh