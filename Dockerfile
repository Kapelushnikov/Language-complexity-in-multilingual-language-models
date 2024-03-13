FROM python:3.10-slim

ENV DATA_ROOT /data
ENV PROJECT_ROOT /thesis

RUN mkdir -p $DATA_ROOT
RUN mkdir -p $PROJECT_ROOT

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git\
    && rm -rf /var/lib/apt/lists/*

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install --no-cache-dir -r requirements.txt

RUN cd $DATA_ROOT git clone https://github.com/tareknaous/readme.git

# Задаем команду по умолчанию
CMD ["python", "train.py"]