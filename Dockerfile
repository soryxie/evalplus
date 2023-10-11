# base env: py38 ubuntu20.04
FROM python:3.8-slim-buster

# install git
RUN apt-get update && apt-get install -y git

# upgrade to latest pip
RUN pip install --upgrade pip

COPY requirements-tools.txt requirements-llm.txt requirements.txt /requirements/

RUN cd /requirements && pip install -r requirements-tools.txt \
    && pip install -r requirements-llm.txt \
    && pip install -r requirements.txt