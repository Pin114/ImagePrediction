# 1. Choose a base image. We'll use a TensorFlow-specific image that includes GPU support.
FROM tensorflow/tensorflow:2.15.0-gpu

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy your requirements file first to leverage Docker's caching.
# This is a best practice to speed up future builds.
COPY requirement.txt ./

# 4. Install the necessary Python packages.
RUN pip install --no-cache-dir -r requirement.txt

# 5. Copy your entire project code into the container's working directory.
# This includes your main Python script and any other files you have.
COPY . .

# 6. Expose any ports if your application were a web service (not needed for your current script,
# but good practice for future web app deployments).
# EXPOSE 5000

# 7. Define the command to run your script. This will execute when the container starts.
CMD ["python", "CNN_Model.py"]