FROM python:3.10.13

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
