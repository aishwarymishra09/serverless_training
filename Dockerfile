FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# SHELL ["/bin/bash", "-c"]

# Update and upgrade the system packages (Worker Template)
# Install missing dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils zstd python3.10-venv git-lfs unzip && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash




# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt

COPY builder/test.py /test.py
RUN python /test.py
RUN rm /test.py


# Fetch the model
# COPY builder/model_fetcher.py /model_fetcher.py
# RUN python /model_fetcher.py
# RUN rm /model_fetcher.py


# Add src files (Worker Template)
ADD src .

ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DEBIAN_FRONTEND noninteractive

ENV PYTHONUNBUFFERED=1
CMD python -u rp_handler.py
