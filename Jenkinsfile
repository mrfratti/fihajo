pipeline {
    agent any

    parameters {
        choice(name: 'ACTION', choices: ['Standard Training', 'Adversarial Training'], description: 'Choose the training mode')
        string(name: 'EPOCHS', defaultValue: '10', description: 'Number of epochs for training')
        string(name: 'BATCH_SIZE', defaultValue: '64', description: 'Batch size for training')
        string(name: 'SAVE_PATH', defaultValue: 'data/models/model.weights.h5', description: 'Path to save model weights')
        string(name: 'EPSILON', defaultValue: '0.3', description: 'Epsilon value for adversarial training or evaluation')
        choice(name: 'OPTIMIZER', choices: ['adadelta', 'adam', 'sgd'], description: 'Optimizer for training')
        choice(name: 'DATASET', choices: ['mnist', 'cifar10', 'fashion_mnist'], description: 'Dataset for training and evaluation')
        booleanParam(name: 'ADV_EVAL', defaultValue: false, description: 'Perform adversarial attacks during evaluation')
    }
    

    stages {
        stage('Setup') {
            steps {
                checkout scm
                sh "mkdir -p data/plots/training data/models data/logs report/reports"
            }
        }

        stage('Install Dependencies') {
            steps {
                timestamps { echo ">>>>>>>>>>Installing dependencies>>>>>>>>>>"}
                sh '''
                pip3 install --upgrade pip
                pip3 install -r requirements.txt
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                timestamps { echo ">>>>>>>>>>Running Unit Tests>>>>>>>>>>"}
                sh '''
                python3 -m unittest discover -v -s ./tests
                '''
            }
        }

        stage('Security Scan') {
            steps {
                timestamps { echo ">>>>>>>>>>Running bandit on source code>>>>>>>>>>"}
                sh '''
                pip3 install bandit
                bandit -r src/ -c bandit.yaml
                '''
            }
        }

        stage('Dependency Security Check') {
            steps {
                sh '''
                pip3 install safety
                safety check
                '''
            }
        }

        stage('Train') {
            when { expression { params.ACTION == 'Standard Training' } }
            steps {
                sh "python3 -m src.cli.main train --dataset ${params.DATASET} --epochs ${params.EPOCHS} --batch ${params.BATCH_SIZE} --save-path ${params.SAVE_PATH} --optimizer ${params.OPTIMIZER}"
            }
        }

        stage('Adversarial Training') {
            when { expression { params.ACTION == 'Adversarial Training' } }
            steps {
                sh "python3 -m src.cli.main train --adv --dataset ${params.DATASET} --epochs ${params.EPOCHS} --batch ${params.BATCH_SIZE} --save-path ${params.SAVE_PATH} --optimizer ${params.OPTIMIZER} --eps ${params.EPSILON}"
            }
        }

        stage('Evaluate') {
            steps {
                script {
                    if (params.ADV_EVAL) {
                        sh "python3 -m src.cli.main evaluate --adv-eval --dataset ${params.DATASET} --model-path ${params.SAVE_PATH} --eps ${params.EPSILON}"
                    } else {
                        sh "python3 -m src.cli.main evaluate --dataset ${params.DATASET} --model-path ${params.SAVE_PATH}"
                    }
                }
            }
        }

        stage('Analyze') {
            steps {
                sh "python3 -m src.cli.main analyze --dataset ${params.DATASET} --model-path ${params.SAVE_PATH}"
            }
        }

        stage('Generate HTML Report') {
            steps {
                sh "python3 -m src.cli.main report"
                publishHTML target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'src/report/reports/',
                    reportFiles: 'index.html',
                    reportName: "HTML Report"
                ]
            }
        }


        stage('Generate HTML Report 2') {
            steps {
                sh "python3 -m src.cli.main reportInteractive"
                publishHTML target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: '',
                    reportFiles: 'test_index.html',
                    reportName: "HTML Report 2"
                ]
            }
        }

    }

    post {
        always {
            cleanWs(
            notFailBuild: true,
            deleteDirs: true,
            patterns: [[pattern: '**', excludes: '**/models/**']]
            )
        }
    }
}
