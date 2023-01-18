# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY test_environment.py test_environment.py
COPY environment.yml environment.yml
COPY src_2/ src_2/
COPY data/ data/
COPY outputs/ outputs/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

RUN ["pip", "install", "-e", "."]
RUN ["pip", "install", "wandb"]
RUN ["pip", "install", "torch-sparse"]
RUN ["pip", "install", "torch-scatter"]

ENTRYPOINT ["python", "-u", "src_2/models/train_model.py"]

