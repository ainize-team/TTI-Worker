FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# set environment variables
ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    DEBIAN_FRONTEND=noninteractive  \
    PROJECT_NAME="TTI-Worker"

ENV PATH="$POETRY_HOME/bin:/workspaces/$PROJECT_NAME/.venv/bin:$PATH"

# Install Poetry
# https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    git \
    openssh-client \
    python3 \
    python3-dev \
    python3-pip  && \
    curl -sSL https://install.python-poetry.org | python3 -