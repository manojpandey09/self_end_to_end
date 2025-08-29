FROM python:3.8-slim-buster 
###fetching this from docker hub

WORKDIR /service
COPY requirements.txt .
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT [ "python3" , "app.py" ]