FROM continuumio/miniconda3:latest

# Install OpenJDK required for PySpark
RUN apt-get update && \
  apt-get install -y openjdk-17-jdk-headless --no-install-recommends && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
# Set JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-1.17.0-openjdk-amd64

WORKDIR /usr/app/src/training

COPY src/training/environment.yml /usr/app/src/training/

# Install dependencies from conda environment file
RUN conda env create -f environment.yml --name training

ARG DATABRICKS_PASSWORD

# Create Databricks authentication file
RUN echo "[DEFAULT]" > ~/.databrickscfg && \
    echo "host = https://community.cloud.databricks.com/" >> ~/.databrickscfg && \
    echo "username = correovmp@gmail.com" >> ~/.databrickscfg && \
    echo "password = ${DATABRICKS_PASSWORD}" >> ~/.databrickscfg

# Make RUN commands use the conda environment:
SHELL ["conda", "run", "-n", "training", "/bin/bash", "-c"]

# Copy the preprocessed data used for training
COPY data/preprocessed/ /usr/app/data/preprocessed/

# Alternatively set the data folder as a volume
# VOLUME /usr/app/data/preprocessed/

# Copy the rest of the code
COPY src/training/ /usr/app/src/training/

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility


WORKDIR /usr/app/


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "training", "python", "-m", "src.training.training_pipeline"]

# Alternative to execute the training pipeline interactively. Needs to activate the conda environment first
# CMD ["bash"]

