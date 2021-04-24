FROM python:3.9.2-slim-buster AS base

ARG APP_DIR=/usr/app/

USER root

RUN mkdir ${APP_DIR}

WORKDIR ${APP_DIR}

RUN apt-get update && apt-get install -y build-essential 

#graphviz is required by prefect[viz] v.0.14.12
RUN apt-get -y install graphviz

COPY requirements.txt ${APP_DIR}

RUN pip3 install -r requirements.txt