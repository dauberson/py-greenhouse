FROM python:3.9.2-slim-buster AS base

ARG APP_DIR=/usr/app/

USER root

RUN mkdir ${APP_DIR}

WORKDIR ${APP_DIR}

#graphviz is required by prefect[viz] v.0.14.12
RUN apt-get update \
    && apt-get install -y build-essential graphviz \
    && apt-get clean

COPY requirements.txt ${APP_DIR}

RUN pip install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

RUN python3 -m spacy download pt_core_news_sm
