FROM python:3.10-slim

ENV PROJECT_ROOT /thesis
ENV DATA_ROOT /thesis/data

RUN mkdir -p $DATA_ROOT
RUN mkdir -p $PROJECT_ROOT

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl\
    && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

COPY . $PROJECT_ROOT
COPY ./data $DATA_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install --no-cache-dir -r requirements.txt

RUN pre-commit install
RUN pre-commit autoupdate
RUN dvc init

RUN cd $DATA_ROOT && git clone https://github.com/tareknaous/readme.git
