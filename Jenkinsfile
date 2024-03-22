pipeline {
    agent any

    stages {
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

                    def save_path = input(
                        id: 'userInput',
                        message: 'Save path:',
                        defaultValue: null,
                        parameters: [
                            string(description: 'Save path', name: 'SavePath')
                        ]
                    )
    
                    sh "python -m src.cli.main train --epochs ${epochs} --batch ${batch_size} --dataset mnist --save-path ${save_path}"

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
                sh 'python -m src.cli.main evaluate --dataset mnist --model-path ${load_path}'

                def load_path = input(
                    id: 'userInput',
                    message: 'Load path',
                    defaultValue: null,
                    parameters: [
                        string(description: 'Path to weights', name:'LoadPath')
                    ]
                )
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
