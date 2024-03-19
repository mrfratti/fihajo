pipeline {
    agent any

    stages {
        stage('CONNECTION') {
            steps {
                checkout([$class: 'GitSCM', 
                    branches: [[name: 'refs/heads/uncertainty_analysis_2']],
                    extensions: [], 
                    userRemoteConfigs: [[
                        url: 'git@github.com:mrfratti/fihajo.git',
                        credentials: 'own ssh code'
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
        
        stage('RUN') {
            steps {
                sh 'python -m src.cli.main'
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
    
                    sh "python -m src.cli.main train --epochs ${epochs} --batch ${batch_size}"
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
    
                    sh "python -m src.cli.main train --epochs ${epochs} --batch ${batch_size}"
                }
            }
        }
        
        
        stage('EVALUATE') {
            steps {
                sh 'python -m src.cli.main evaluate'
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