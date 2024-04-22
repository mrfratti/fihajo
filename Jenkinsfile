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

                    def user_input = input(
                        id: 'user_input', 
                        message: 'Select input option:', 
                        parameters: [
                            choice(
                                name: 'choices',
                                choices: ['Custom input', 'Recommended input'],
                                description: ''
                            )
                        ]
                    )

                    def retry_count = 1
                    def retry_interval = 5
                    def command_output

                    for (int i = 0; i < retry_count; i++) {

                        if (user_input == 'Custom input') {
                            echo 'Executing Custom input'

                            def defaultfile = "data/models/mnist.model.h5"
                            def fullpath ="${env.WORKSPACE}/${defaultfile}"
                            def epochs = input(id: 'userInputEpochs', message: 'Enter the number of epochs:', parameters: [string(defaultValue: '10', description: 'Number of epochs', name: 'epochs')])
                            def batch_size = input(id: 'userInputBatchSize', message: 'Enter the batch size:', parameters: [string(defaultValue: '32', description: 'Batch size', name: 'batch')])
                            def save_path = input(id: 'userInputSavePath', message: 'Enter the save path for model weights:', parameters: [string(defaultValue:fullpath, description: 'Model save path', name: 'savePath')])

                            def command_text = "echo | python -m src.cli.main --verbose train --dataset mnist --epochs ${epochs} --batch ${batch_size} --save-path ${save_path}"
                            command_output = sh(script: command_text, returnStdout: true, returnStatus: true)
                        }
                        else if (user_input == 'Recommended input') {
                            echo 'Executing recommended input'
                            def command_text = "echo | python -m src.cli.main --config src/cli/config/train.json --verbose"
                            command_output = sh(script: command_text, returnStdout: true, returnStatus: true)
                        }

                        echo "TEST 2 ..."
                        if (command_output == 0) {
                            echo "Running successfully! Next Stage ..."
                            break
                        } else {
                            echo "Error found! Retrying in ${retry_interval} sec ..."
                            sleep retry_interval
                        }

                    }

                    echo  "TEST 3 ..."
                    // Display error output, linked up with python CLI error output
                    if (command_output != 0) {
                            echo "Error output:"
                            def terminal_lines = currentBuild.rawBuild.getLog(1000)
                            def terminal_error = terminal_lines.findAll { line -> line.contains("error:") }
                            if (!terminal_error.isEmpty()) {
                                def terminal_last_line = terminal_error.last()
                                echo "Last error output:"
                                echo terminal_last_line
                            } else {
                                echo "No error!"
                            }

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
                        def defaultfile = "data/models/model.weights.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main --verbose evaluate --dataset mnist"
                    }

                    else if (env.stage_choice == 'Adverserial') {
                        def defaultfile = "data/models/adv_model.weights.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main --verbose evaluate --dataset mnist "
                    }

                }
            }
        }
        stage('ANALYZE') {
            steps {
                script {

                    if (env.stage_choice == 'Train') {
                        def defaultfile = "data/models/model.weights.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main analyze --dataset mnist "
                    }

                    else if (env.stage_choice == 'Adverserial') {
                        def defaultfile = "data/models/adv_model.weights.h5"
                        def fullpath ="${env.WORKSPACE}/${defaultfile}"
                        sh "echo | python -m src.cli.main analyze --dataset mnist "
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
