# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code
COPY . .

# Set permissions for Hugging Face (Port 7860 is required)
RUN chmod -R 777 /code

# Start both FastAPI (backend) and Streamlit (frontend) 
# Note: Adjust 'main:app' to your FastAPI filename and 'app.py' to your Streamlit filename
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 7860 --server.address 0.0.0.0