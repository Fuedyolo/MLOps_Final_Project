FROM python:3.9-slim

EXPOSE 80
COPY ./app /code/app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
COPY setup.py setup.py
COPY test_environment.py test_environment.py
COPY environment.yml environment.yml
COPY src_2/ src_2/
COPY data/ data/
COPY outputs/ outputs/

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN ["pip", "install", "-e", "."]
#RUN ["pip", "install", "torch-sparse"]
#RUN ["pip", "install", "torch-scatter"]




CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]