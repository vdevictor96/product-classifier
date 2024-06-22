FROM continuumio/miniconda3:latest

WORKDIR /usr/app/src/inference

COPY src/inference/environment.yml /usr/app/src/inference/

# Install dependencies from conda environment file
RUN conda env create -f environment.yml --name inference

ARG DATABRICKS_PASSWORD

# Create Databricks authentication file
RUN echo "[DEFAULT]" > ~/.databrickscfg && \
    echo "host = https://community.cloud.databricks.com/" >> ~/.databrickscfg && \
    echo "username = correovmp@gmail.com" >> ~/.databrickscfg && \
    echo "password = ${DATABRICKS_PASSWORD}" >> ~/.databrickscfg

# Make RUN commands use the conda environment:
SHELL ["conda", "run", "-n", "inference", "/bin/bash", "-c"]

# Copy the code
COPY src/inference/ /usr/app/src/inference/

# Expose port 8000 for the application
EXPOSE 8000

WORKDIR /usr/app/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "inference", "python", "-m", "src.inference.api"]

# CMD ["bash"]

