# Jenkins based Command-Line Tool for AI Model Uncertainty Analysis

## Overview 

This repository contains the AI Model Uncertainty Analysis Tool, a command-line tool developed as part of a bachelor's thesis. The tool is designed to access the reliability of AI model predictions by analyzing the uncertainty of the models. It generates informative reports, including histograms of uncertainty distribution, highlights highly uncertain inputs, their corresponding uncertainty values, and the model's predictions. 

[![Pylint](https://github.com/mrfratti/fihajo/actions/workflows/pylint.yml/badge.svg)](https://github.com/mrfratti/fihajo/actions/workflows/pylint.yml)
[![Python unittest CI](https://github.com/mrfratti/fihajo/actions/workflows/python-test.yml/badge.svg)](https://github.com/mrfratti/fihajo/actions/workflows/python-test.yml)
## Features 

- **Uncertainty Quantification:** 
- **Visualization:** Visualizes the uncertainty distribution as histograms.
- **High Uncertainty Identification:** Identifies and highlights inputs with high uncertainty, displaying their uncertainty values and model predictions.
- **Report Generation:**

## Installation

The tool is containerized using [Docker](https://www.docker.com), ensuring a consistent and isolated environment. The Docker image includes [Jenkins](https://www.jenkins.io) with Python, TensorFlow, and the Uncertainty Wizard pre-installed.


## Building the Docker Image 
```bash
docker build -t fihajo .
```
## Create and Run the Docker Image
```bash
docker run -p 8080:8080 fihajo
```
## Running the Docker Image
```bash
docker start image-name
```
## Usage 



## Development

This project is set up with Jenkins for continuous integration and deployment. 


## Setting Up the Development Environment

...

## License

Distributed under the MIT License. See `LICENSE` for more information. 

# Contact

- Firat Celebi
- Joakim Hole Polden
- Harykaran Lambotharan
