# PDX Analysis Tutorial Docker Environment
FROM continuumio/miniconda3:latest

LABEL maintainer="PDX Analysis Tutorial"
LABEL description="Complete environment for PDX (Patient-Derived Xenograft) analysis"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml /app/environment.yml

# Create conda environment
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "pdx_analysis", "/bin/bash", "-c"]

# Install additional R packages that might not be in conda
RUN conda run -n pdx_analysis Rscript -e "\
  if (!require('configr')) install.packages('configr', repos='https://cran.r-project.org'); \
  if (!require('logger')) install.packages('logger', repos='https://cran.r-project.org'); \
  if (!require('broom.mixed')) install.packages('broom.mixed', repos='https://cran.r-project.org'); \
  if (!require('ggpubr')) install.packages('ggpubr', repos='https://cran.r-project.org'); \
  if (!require('corrplot')) install.packages('corrplot', repos='https://cran.r-project.org'); \
  if (!require('emmeans')) install.packages('emmeans', repos='https://cran.r-project.org')"

# Copy project files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/logs

# Set permissions
RUN chmod +x /app/tests/integration_test.sh
RUN chmod +x /app/setup.sh

# Expose Jupyter port
EXPOSE 8888

# Set default command
CMD ["conda", "run", "--no-capture-output", "-n", "pdx_analysis", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Alternative: Run bash for interactive use
# CMD ["conda", "run", "--no-capture-output", "-n", "pdx_analysis", "/bin/bash"]