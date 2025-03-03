FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
