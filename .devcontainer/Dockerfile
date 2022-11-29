FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# set environment variables
ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive  \
    CONDA_MIRROR=https://github.com/conda-forge/miniforge/releases/latest/download  \
    CONDA_DIR=/opt/conda \
    CONDA_ROOT=/opt/conda

ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Install Ubuntu Package
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    build-essential \
    wget \
    git \
    openssh-client 

RUN set -x && \
    # Miniforge installer
    miniforge_arch=$(uname -m) && \
    miniforge_installer="Mambaforge-Linux-${miniforge_arch}.sh" && \
    wget --quiet "${CONDA_MIRROR}/${miniforge_installer}" && \
    /bin/bash "${miniforge_installer}" -f -b -p "${CONDA_DIR}" && \
    rm "${miniforge_installer}" && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    mamba install --quiet --yes python=3.9 && \
    # Pin major.minor version of python
    mamba list python | grep '^python ' | tr -s ' ' | cut -d ' ' -f 1,2 >> "${CONDA_DIR}/conda-meta/pinned" && \
    # Using conda to update all packages: https://github.com/mamba-org/mamba/issues/1092
    conda update --all --quiet --yes && \
    conda clean --all -f -y