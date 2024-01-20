# Use the official Jenkins image as a parent image
FROM jenkins/jenkins:lts

# Switch to root to install Python, TensorFlow, and uncertainty_wizard
USER root

# Install Python, pip, and the venv module
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install TensorFlow and uncertainty_wizard within the virtual environment
RUN pip install --no-cache-dir tensorflow uncertainty_wizard

# Switch back to the jenkins user
USER jenkins

