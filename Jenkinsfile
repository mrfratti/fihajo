pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                checkout scm
                
                //create directories
                dir('data/plots/training'){writeFile file: "temp", text =""}
                dir('data/models'){writeFile file: "temp", text =""}
                dir('data/logs'){writeFile file: "temp", text =""}
                dir('report/reports'){writeFile file: "temp", text =""}
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

                    sh "python -m src.cli.main train --dataset mnist --epochs ${epochs} --batch ${batch_size} --save-path ${save_path}"
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
                    sh "python -m src.cli.main evaluate --dataset mnist --model-path ${load_path}"
                }
            }
        }

        stage('HTML Report') {
            steps {
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
