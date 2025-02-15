# Use the official Python image with PyTorch support
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5002

# Run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5002", "app:app"]
