FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y software-properties-common \
 && apt-add-repository -y ppa:neovim-ppa/unstable

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl \
    fish \
    git \
    graphviz \
    htop \
    neovim \
    python3-dev \
    python3-venv \
    tmux \
    wget \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

# install poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN chmod 755 ${POETRY_HOME}/bin/poetry

CMD ["/bin/bash"]
