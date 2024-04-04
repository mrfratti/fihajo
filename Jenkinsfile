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

        stage('Train or Attack') {
            steps {
                script {
                    env.stage_choice = input(id: 'userInput',
                                             message: 'Choose action:',
                                             parameters: [choice(name: 'STAGE OPTION',
                                                                 choices: ['Train', 'Attack'],
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

        stage('ATTACK') {
            when {
                expression { return env.stage_choice == 'Attack' }
            }
            steps {
                echo 'Running Attack...'
                script {
                    sh "echo Attack stage is not yet implemented"
                }
            }
        }

        stage('EVALUATE') {
            steps {
                script {
                    def defaultfile = "data/models/mnist.model.h5"
                    def fullpath ="${env.WORKSPACE}/${defaultfile}"
                    def load_path = input(id: 'userInputLoadPath', message: 'Enter the load path for model weights:', parameters: [string(description: 'Model load path', name: 'loadPath', defaultValue: fullpath)])
                    sh "echo | python -m src.cli.main --verbose evaluate --dataset mnist --report"
                }
            }
        }

        stage('HTML Report') {
            steps {
                publishHTML( target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'report/reports/',
                    reportFiles: 'index.html',
                    reportName: "HTML Report"
                ]
    
            script {
                def reportURL = "${env.BUILD_URL}HTMLReport/"
                echo "<a href='${reportURL}' target='_blank'>Open HTML Report</a>"
            }
            
            )
        }}
    }
})
