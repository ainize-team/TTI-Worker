FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# set environment variables
ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PROJECT_NAME="TTI-Worker" \
    DEBIAN_FRONTEND=noninteractive

# Generate and Set locals
RUN \
    apt-get update && \
    apt-get install -y locales && \
    # install locales-all?
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LC_ALL="en_US.UTF-8" \
    LANG="en_US.UTF-8" \
    LANGUAGE="en_US:en"

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

ENV PATH="$POETRY_HOME/bin:/workspaces/$PROJECT_NAME/.venv/bin:$PATH"

# Install Poetry
# https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions
RUN curl -sSL https://install.python-poetry.org | python3.9 - 