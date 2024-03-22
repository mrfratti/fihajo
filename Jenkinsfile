pipeline {
    agent any

    stages {
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
        
        stage('RUN') {
            steps {
                 sh 'python -m src.cli.main --config src/cli/config/train.json'
            }
        }
        
        
        
        stage('TRAIN OR ATTACK') {
            steps {
                script {
                    env.stage_choice = input(id: 'userInput', 
                    message: 'Choose:', 
                    parameters: [choice(name: 'STAGE OPTION:', 
                    choices: ['Train', 'Attack'], 
                    description: 'Select stage!')])
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
                    def epochs = input(
                        id: 'userInput',
                        message: 'Number of epochs:',
                        defaultValue: '10',
                        parameters: [
                            string(description: 'Number of epochs', name: 'Epochs')
                        ]
                    )
    
                    def batch_size = input(
                        id: 'userInput',
                        message: 'Batch size:',
                        defaultValue: '1000',
                        parameters: [
                            string(description: 'Batch size', name: 'BatchSize')
                        ]
                    )
    
                    sh "python -m src.cli.main train --epochs ${epochs} --batch ${batch_size} --dataset mnist"
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
                    def epochs = input(
                        id: 'userInput',
                        message: 'Number of epochs:',
                        defaultValue: '10',
                        parameters: [
                            string(description: 'Number of epochs', name: 'Epochs')
                        ]
                    )
    
                    def batch_size = input(
                        id: 'userInput',
                        message: 'Batch size:',
                        defaultValue: '1000',
                        parameters: [
                            string(description: 'Batch size', name: 'BatchSize')
                        ]
                    )
    
                    sh "python -m src.cli.main train --epochs ${epochs} --batch ${batch_size} --dataset mnist"
                }
            }
        }
        
        
        stage('EVALUATE') {
            steps {
                sh 'python -m src.cli.main --evaluate --dataset mnist'
            }
        }
        
        stage('HTML Report') {
            steps {
                publishHTML target: [
                    allowMissing: true,
                    alwaysLinkToLastBuild: false,
                    keepAll: true,
                    reportDir: 'reports',
                    reportFiles: 'report_1.html',
                    reportName: "HTML Report"
                ]
            }
        }
    }
}
