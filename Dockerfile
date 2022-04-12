FROM python:3.7-slim
WORKDIR /usr/src/app
ADD requirements.txt .
RUN pip3 install -r requirements.txt
ADD . .
ENTRYPOINT ["python", "-u", "main.py"]