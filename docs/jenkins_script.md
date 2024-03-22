pipeline {
    agent any

    stages {
        stage('CONNECTION') {
            steps {
                checkout([$class: 'GitSCM', 
                    // branches: [[name: 'refs/heads/uncertainty_analysis_2']],
                    branches: [[name: 'refs/heads/main']],
                    extensions: [], 
                    userRemoteConfigs: [[
                        url: 'git@github.com:mrfratti/fihajo.git',
                        credentials: 'ssh code'
                    ]]
                ])
            }
        }
        


        stage('INFO') {
            steps {
                script {
                    echo 'src directory:'
                    sh 'ls -la src/'
                    
                    echo 'src/cli directory:'
                    sh 'ls -la src/cli/'
                }
            }
        }
    
        
        
        stage('Train or Attack') {
            steps {
                script {
                    env.stage_choice = input(id: 'userInput',
                                             message: 'Choose action:',
                                             parameters: [choice(name: 'STAGE OPTION',
                                                                 choices: ['Train', 'Train_Default', 'Attack', 'Attack_Default'],
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
                    def epochs = input(id: 'userInputEpochs', message: 'Enter the number of epochs:', parameters: [string(defaultValue: '10', description: 'Number of epochs', name: 'epochs')])
                    def batch_size = input(id: 'userInputBatchSize', message: 'Enter the batch size:', parameters: [string(defaultValue: '32', description: 'Batch size', name: 'batch')])
                    def save_path = input(id: 'userInputSavePath', message: 'Enter the save path for model weights:', parameters: [string(defaultValue: 'models/mnist_model.h5', description: 'Model save path', name: 'savePath')])

                    sh "python -m src.cli.main train --dataset mnist --epochs ${epochs} --batch ${batch_size} --save-path ${save_path}"
                }
            }
        }
        


        stage('TRAIN DEFAULT') {
            when {
                expression { return env.stage_choice == 'Train_Default' }
            }
            steps {
                echo 'Running Train...'
                script {
                    def epochs = input(id: 'userInputEpochs', message: 'Enter the number of epochs:', parameters: [string(defaultValue: '10', description: 'Number of epochs', name: 'epochs')])
                    def batch_size = input(id: 'userInputBatchSize', message: 'Enter the batch size:', parameters: [string(defaultValue: '32', description: 'Batch size', name: 'batch')])
                    def save_path = input(id: 'userInputSavePath', message: 'Enter the save path for model weights:', parameters: [string(defaultValue: 'models/mnist_model.h5', description: 'Model save path', name: 'savePath')])

                    sh "python -m src.cli.main train /src/cli/config/train.json"
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
        


        stage('ATTACK DEFAULT') {
            when {
                expression { return env.stage_choice == 'Train_Default' }
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
                    def load_path = input(id: 'userInputLoadPath', message: 'Enter the load path for model weights:', parameters: [string(description: 'Model load path', name: 'loadPath', defaultValue: 'models/mnist_model.h5')])
                    sh "echo | python -m src.cli.main evaluate --dataset mnist --model-path ${load_path}"
                }
            }
        }



        stage('HTML Report') {
            steps {
                publishHTML target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'reports',
                    reportFiles: 'index.html',
                    reportName: "HTML Report"
                ]
            }
        }
    }
}