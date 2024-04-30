pipeline {
    agent any

    parameters {
        choice(name: 'ACTION', choices: ['Standard Training', 'Adversarial Training'], description: 'Choose the training mode')
        choice(name: 'INPUT_PARAMETERS', choices: ['Custom parameters', 'Recommended parameters'], description: 'Select input configuration')
        string(name: 'EPOCHS', defaultValue: '10', description: 'Number of epochs for training')
        string(name: 'BATCH_SIZE', defaultValue: '64', description: 'Batch size for training')
        string(name: 'SAVE_PATH', defaultValue: '', description: 'Optional: custom path to save model weights')
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
                    if (params.INPUT_PARAMETERS == 'Custom parameters') {
                        env.EPOCHS = params.EPOCHS
                        env.BATCH_SIZE = params.BATCH_SIZE
                        env.SAVE_PATH = params.SAVE_PATH ?: "data/models/model.weights.h5"
                    }
                }
            }
        }

        stage('Train') {
            when { expression { params.ACTION == 'Standard Training' } }
            steps {
                script {
                    if (params.INPUT_PARAMETERS == 'Custom parameters') {
                        sh "echo | python -m src.cli.main train --dataset mnist --epochs ${env.EPOCHS} --batch ${env.BATCH_SIZE} --save-path ${env.SAVE_PATH}"
                    } else {
                        sh "echo | python -m src.cli.main --config src/cli/config/train.json"
                    }
                }
            }
        }

        stage('Adversarial Training') {
            when { expression { params.ACTION == 'Adversarial Training' } }
            steps {
                script {
                    if (params.INPUT_PARAMETERS == 'Custom parameters') {
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
                    def modelPath = params.SAVE_PATH ?: "data/models/model.weights.h5"
                    sh "python -m src.cli.main evaluate --dataset mnist --model-path ${modelPath}"
                }
            }
        }

        stage('Analyze') {
            steps {
                script {
                    def modelPath = params.SAVE_PATH ?: "data/models/model.weights.h5"
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
