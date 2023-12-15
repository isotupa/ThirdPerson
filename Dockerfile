# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies needed by your Python project
RUN pip install -r requirements.txt

# Set the environment variable to allow GUI applications to connect to the X server
ENV DISPLAY=:0

# Command to run your application (replace with your command)
CMD ["python", "main.py"]
