pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                checkout scm
                
                //create directories
                dir('data/plots/training'){sh 'pwd -P'}
                dir('data/models'){sh 'pwd -P'}
                dir('data/logs'){sh 'pwd -P'}
                dir('report/reports'){sh 'pwd -P'}
            }
        }

        stage('Train or Adverserial Training') {
            steps {
                script {
                    env.stage_choice = input(id: 'userInput',
                                             message: 'Choose action:',
                                             parameters: [choice(name: 'STAGE OPTION',
                                                                 choices: ['Train', 'Adverserial'],
                                                                 description: 'Select stage')])
                }
            }
        }

        stage('TRAIN') {
            when {
                expression { return env.stage_choice == 'Train' }
            }
            steps {
                echo 'Running Train...'
                script {
                    def defaultfile = "data/models/mnist.model.h5"
                    def fullpath ="${env.WORKSPACE}/${defaultfile}"
                    def epochs = input(id: 'userInputEpochs', message: 'Enter the number of epochs:', parameters: [string(defaultValue: '10', description: 'Number of epochs', name: 'epochs')])
                    def batch_size = input(id: 'userInputBatchSize', message: 'Enter the batch size:', parameters: [string(defaultValue: '32', description: 'Batch size', name: 'batch')])
                    def save_path = input(id: 'userInputSavePath', message: 'Enter the save path for model weights:', parameters: [string(defaultValue:fullpath, description: 'Model save path', name: 'savePath')])

                    sh "echo | python -m src.cli.main --verbose train --dataset mnist --epochs ${epochs} --batch ${batch_size} --save-path ${save_path}"
                }
            }
        }

        stage('ADVERSERIAL TRAINING') {
            when {
                expression { return env.stage_choice == 'Adverserial' }
            }
            steps {
                echo 'Running Adverserial....'
                script {
                    sh "echo | python -m src.cli.main --verbose train --adv --dataset mnist"
                }
            }
        }

        stage('EVALUATE') {
            steps {
                script {

                    if (env.stage_choice == 'Train') {
                        def defaultfile = "./data/models/mnist.model.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main --verbose evaluate --dataset mnist --model-path ${defaultfile}"
                    }

                    else if (env.stage_choice == 'Adverserial') {
                        def defaultfile = "./data/models/adv_model.weights.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main --verbose evaluate --dataset mnist --model-path ${defaultfile}"
                    }

                }
            }
        }
        stage('ANALYZE') {
            steps {
                script {

                    if (env.stage_choice == 'Train') {
                        def defaultfile = "./data/models/mnist.model.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main analyze --dataset mnist --model-path ${defaultfile}"
                    }

                    else if (env.stage_choice == 'Adverserial') {
                        def defaultfile = "./data/models/adv_model.weights.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main analyze --dataset mnist --model-path ${defaultfile}"
                    }

                }
            }
        }

        stage('HTML REPORT') {
            steps {
                script{
                    sh "echo | python -m src.cli.main report"
                }
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
