FROM python:3.9

WORKDIR /usr/src/app

COPY . .
COPY requirements.txt ./

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#RUN pip install cmake
#RUN pip install dlib
RUN pip install mysqlclient

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
EXPOSE 8000