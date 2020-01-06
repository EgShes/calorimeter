FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

ENV DEVICE=cpu
RUN apt-get update -y
COPY . ./app
WORKDIR ./app
RUN pip3 install -r requirements.txt
