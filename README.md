# Jenkins based Command-Line Tool for AI Model Uncertainty Analysis

## Overview 

This repository contains the AI Model Uncertainty Analysis Tool, a command-line tool developed as part of a bachelor's 
thesis. The tool is designed to access the reliability of AI model predictions by analyzing the uncertainty of the 
models. It generates informative reports, including histograms of uncertainty distribution, highlights highly 
uncertain inputs, their corresponding uncertainty values, and the model's predictions. The primary libraries used 
for this project are Keras/TensorFlow, Uncertainty Wizard and CleverHans libraries.

[![Pylint](https://github.com/mrfratti/fihajo/actions/workflows/pylint.yml/badge.svg)](https://github.com/mrfratti/fihajo/actions/workflows/pylint.yml)
[![Python unittest CI](https://github.com/mrfratti/fihajo/actions/workflows/python-test.yml/badge.svg)](https://github.com/mrfratti/fihajo/actions/workflows/python-test.yml)
## Features 

- **Training:** Standard and adversarial training modes.
- **Evaluation:** Standard and adversarial evaluations of models.
- **Uncertainty Analysis:** Quantifies and visualizes model uncertainty.
- **Report Generation:** Generates detailed reports with interactive visualizations.
- **Jenkins Pipeline:** Automates the workflow using Jenkins.

## Setup Instructions
#### 1. Clone the Repository
```sh
git clone https://github.com/mrfratti/fihajo.git
cd ai-uncertainty-analysis
```

#### 2. Install Dependencies
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage Instructions
#### Command-Line Interface
Below are some example commands to get you started.

**Training**
```sh
python -m src.cli.main train --dataset mnist
```

**Adversarial Training**
```sh
python -m src.cli.main train --adv --eps 0.3 --dataset mnist 
```
**Evaluation**
```sh
python -m src.cli.main evaluate --dataset mnist
```

**Adversarial Evaluation**
```sh
python -m src.cli.main evaluate --adv-eval --dataset mnist
```

**Uncertainty Analysis**
```sh
python -m src.cli.main analyze --dataset mnist
```

**Report Generation**
```sh
python -m src.cli.main report
```

## Jenkins Pipeline
The Jenkins pipeline automates training, evaluation, analysis and report generation processes. 
Below is an overview of the stages included in the Jenkinsfile:

1. **Setup:** Checkout code and create necessary directories.
2. **Install Dependencies:** Install required Python packages.
3. **Unit Tests:** Run unit tests to ensure code quality.
4. **Security Scan:** Scan source code for security issues using Bandit
5. **Dependency Security Check:** Check for security vulnerabilities in dependencies using Safety.
6. **Train:** Run standard or adversarial training based on parameters.
7. **Evaluate:** Evaluate the trained model.
8. **Analyze:** Analyze model uncertainty.
9. **Generate HTML Report:** Generate and publish HTML reports.

## Docker Setup
If you want to run Jenkins in a Docker container, use the provided Dockerfile. This setup isolates the Jenkins 
environment and pre-installs necessary dependencies.

#### Build the Docker Image
```sh
docker build -t jenkins
```

#### Run the Docker Container
```sh
docker run -p 8080:8080 -p 50000:50000 jenkins
```

## Environment Variable Setup for Jenkins

Refer to the [Jenkins Environment Variable Setup Guide](docs/jenkins_path_tutorial.md) for detailed instructions on 
configuring environment variables in Jenkins to ensure smooth execution of your build process. 



## License

Distributed under the MIT License. See `LICENSE` for more information. 

# Contact

- Firat Celebi
- Joakim Hole Polden
- Harykaran Lambotharan
