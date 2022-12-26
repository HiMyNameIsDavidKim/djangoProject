FROM python:3.9

WORKDIR /usr/src/app

COPY . .
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#RUN pip install cmake
#RUN pip install dlib
RUN pip install scikit-learn
RUN pip install mysqlclient

CMD ["bin/sh", "-c", "python", "manage.py", "runserver", "0.0.0.0:8000"]
EXPOSE 8000