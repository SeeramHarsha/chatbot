FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY . .
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]
RUN fallocate -l 512M /swapfile && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile && \
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
