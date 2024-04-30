pipeline {
    agent any

    parameters {
        choice(name: 'ACTION', choices: ['Train', 'Adversarial'], description: 'Choose action for the pipeline')
        choice(name: 'INPUT_TYPE', choices: ['Custom input', 'Recommended input'], description: 'Select input option')
        string(name: 'EPOCHS', defaultValue: '10', description: 'Enter the number of epochs')
        string(name: 'BATCH_SIZE', defaultValue: '64', description: 'Enter the batch size')
        string(name: 'SAVE_PATH', defaultValue: '', description: 'Enter the save path for model weights (leave blank for default)')
    }

    stages {
        stage('Setup') {
            steps {
                checkout scm
                sh "mkdir -p data/plots/training data/models data/logs report/reports"
            }
        }

        stage('Configure Input') {
            steps {
                script {
                    // Configure environment variables based on user input
                    if (params.INPUT_TYPE == 'Custom input') {
                        env.EPOCHS = params.EPOCHS
                        env.BATCH_SIZE = params.BATCH_SIZE
                        env.SAVE_PATH = params.SAVE_PATH ?: "data/models/default_model.weights.h5"
                    }
                }
            }
        }

        stage('Train') {
            when { expression { params.ACTION == 'Train' } }
            steps {
                script {
                    def command = params.INPUT_TYPE == 'Custom input' ?
                        "python -m src.cli.main train --dataset mnist --epochs ${env.EPOCHS} --batch ${env.BATCH_SIZE} --save-path ${env.SAVE_PATH}" :
                        "python -m src.cli.main --config src/cli/config/train.json --verbose"
                    sh command
                }
            }
        }

        stage('Adversarial Training') {
            when { expression { params.ACTION == 'Adversarial' } }
            steps {
                script {
                    if (params.INPUT_TYPE == 'Custom input') {
                        sh "python -m src.cli.main train --adv --dataset mnist --epochs ${env.EPOCHS} --batch ${env.BATCH_SIZE} --save-path ${env.SAVE_PATH}"
                    } else {
                        sh "python -m src.cli.main --config src/cli/config/train_adv.json --verbose"
                    }
                }
            }
        }

        stage('Evaluate') {
            steps {
                script {
                    def modelPath = params.SAVE_PATH ?: "data/models/default_model.weights.h5"
                    sh "python -m src.cli.main evaluate --dataset mnist --model-path ${modelPath}"
                }
            }
        }

        stage('Analyze') {
            steps {
                script {
                    def modelPath = params.SAVE_PATH ?: "data/models/default_model.weights.h5"
                    sh "python -m src.cli.main analyze --dataset mnist --model-path ${modelPath}"
                }
            }
        }

        stage('Generate HTML Report') {
            steps {
                sh "python -m src.cli.main report"
                publishHTML target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'report/reports/',
                    reportFiles: 'index.html',
                    reportName: "HTML Report"
                ]
            }
        }
    }
}
