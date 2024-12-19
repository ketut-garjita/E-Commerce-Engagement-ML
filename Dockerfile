# Use the official TensorFlow Serving image
FROM tensorflow/serving:latest

# Create the model directory at /models
RUN mkdir -p /models/saved_model/1

# Copy the model to the correct directory
COPY ./saved_model/* /models/saved_model/1/

# Set environment variable for TensorFlow Serving to use the model
ENV MODEL_NAME=saved_model
