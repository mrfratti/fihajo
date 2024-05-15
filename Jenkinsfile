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
        booleanParam(name: 'REPORT_INT', defaultValue: false, description: 'Generate Interactive HTML Report')
    }
    

    environment {
        PLOT_DIR = 'data/plots/training'
        MODEL_DIR = 'data/models'
        LOG_DIR = 'data/logs'
        REPORT_DIR = 'report/reports'
    }

    stages {
        stage('Setup') {
            steps {
                script {
                    sh "mkdir -p ${PLOT_DIR} ${MODEL_DIR} ${LOG_DIR} ${REPORT_DIR}"
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                timestamps { echo ">Installing dependencies"}
                sh '''
                pip3 install --upgrade pip
                pip3 install -r requirements.txt
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                timestamps { echo "Running Unit Tests"}
                sh '''
                python3 -m unittest discover -v -s ./tests
                '''
            }
        }

        stage('Security Scan') {
            steps {
                timestamps { echo "Running bandit on source code"}
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
                safety check -r requirements.txt
                '''
            }
        }

        stage('Train') {
            when { expression { params.ACTION == 'Standard Training' } }
            steps {
                script {
                    def trainCmd = "python3 -m src.cli.main train --dataset ${params.DATASET} --epochs ${params.EPOCHS} --batch ${params.BATCH_SIZE} --save-path ${params.SAVE_PATH} --optimizer ${params.OPTIMIZER}"
                    if (params.REPORT_INT) {
                        trainCmd += " --interactive"
                    }
                    sh trainCmd
                }
            }
        }

        stage('Adversarial Training') {
            when { expression { params.ACTION == 'Adversarial Training' } }
            steps {
                script {
                    def advTrainCmd = "python3 -m src.cli.main train --adv --dataset ${params.DATASET} --epochs ${params.EPOCHS} --batch ${params.BATCH_SIZE} --save-path ${params.SAVE_PATH} --optimizer ${params.OPTIMIZER} --eps ${params.EPSILON}"
                    if (params.REPORT_INT) {
                        advTrainCmd += " --interactive"
                    }
                    sh advTrainCmd
                }
            }
        }

        stage('Evaluate') {
            steps {
                script {
                    def evalCmd = "python3 -m src.cli.main evaluate --dataset ${params.DATASET} --model-path ${params.SAVE_PATH}"
                    if (params.ADV_EVAL) {
                        evalCmd += " --adv-eval --eps ${params.EPSILON}"
                    }
                    if (params.REPORT_INT) {
                        evalCmd += " --interactive"
                    }
                    sh evalCmd
                }
            }
        }

        stage('Analyze') {
            steps {
                script {
                    def analyzeCmd = "python3 -m src.cli.main analyze --dataset ${params.DATASET} --model-path ${params.SAVE_PATH}"
                    if (params.REPORT_INT) {
                        analyzeCmd += " --interactive"
                    }
                    sh analyzeCmd
                }
            }
        }

        stage('Generate HTML Report') {
            steps {

                script {
                    if (params.REPORT_INT) {
                        sh "python3 -m src.cli.main report"
                        sh "python3 -m src.cli.main report --interactive"

                        publishHTML target: [
                            allowMissing: false,
                            alwaysLinkToLastBuild: true,
                            keepAll: true,
                            reportDir: "src/report/reports/",
                            reportFiles: "index.html",
                            reportName: "HTML Report"
                        ]
                        publishHTML target: [
                            allowMissing: false,
                            alwaysLinkToLastBuild: true,
                            keepAll: true,
                            reportDir: "./src/report/reports/",
                            reportFiles: "index_interactive.html",
                            reportName: "HTML Interactive Report"
                        ]
                    } 
                    else {
                        sh "python3 -m src.cli.main report"

                        publishHTML target: [
                            allowMissing: false,
                            alwaysLinkToLastBuild: true,
                            keepAll: true,
                            reportDir: "src/report/reports/",
                            reportFiles: "index.html",
                            reportName: "HTML Report"
                        ]
                        
                    }
                }
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
