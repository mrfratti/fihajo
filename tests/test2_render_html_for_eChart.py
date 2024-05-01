import os
import json




def generate_html_with_chart(data_x, data_y1, data_y2, data_y3, data_y4):
    html_content = f"""
    <html>
    <head>

        <title>Interactive Chart</title>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom"></script>

        <style>

            canvas {{
                display: block;
                margin: 0 auto;
            }}

            #controls {{
                text-align: center;
                margin-top: 10px;
            }}

        </style>

    </head>

    <body>

        <canvas id="chart_accuracy_loss" width="800" height="500"></canvas>

        <div id="chart_button">
            <button onclick="data_change_a_l('accuracy')">Show Accuracy</button>
            <button onclick="data_change_a_l('loss')">Show Loss</button>
        </div>

        <script>
            var canvas_accuracy_loss = document.getElementById('chart_accuracy_loss').getContext('2d');
            var chart_data_a_l = new Chart(canvas_accuracy_loss, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(data_x)},
                    datasets: [
                        {{
                            label: 'Accuracy',
                            data: {json.dumps(data_y1)},

                            borderColor: 'rgba(0, 255, 250, 0.8)',
                            backgroundColor: 'rgba(0, 255, 250, 0.8)',
                            borderWidth: 1,
                            pointRadius: 0,
                            yAxisID: 'y',
                        }},
                        {{
                            label: 'Validation Accuracy',
                            data: {json.dumps(data_y2)},

                            borderColor: 'rgba(0, 142, 139, 0.8)',
                            backgroundColor: 'rgba(0, 162, 235, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0,
                            yAxisID: 'y',
                        }},
                        {{
                            label: 'Loss',
                            data: {json.dumps(data_y3)},

                            borderColor: 'rgba(255, 206, 0, 1)',
                            backgroundColor: 'rgba(255, 206, 0, 0.8)',
                            borderWidth: 1,
                            pointRadius: 0,
                            yAxisID: 'y',
                            hidden: true
                        }},
                        {{
                            label: 'Validation Loss',
                            data: {json.dumps(data_y4)},
                            
                            borderColor: 'rgba(0, 192, 192, 1)',
                            backgroundColor: 'rgba(0, 192, 192, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0,
                            yAxisID: 'y',
                            hidden: true
                        }}
                    ]
                }},

                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }},

                    plugins: {{
                        zoom: {{
                            zoom: {{
                                mode: 'xy'
                                wheel: {{ enabled: true }},
                                pinch: {{ enabled: true }},
                            }},
                            pan: {{
                                mode: 'xy'
                                enabled: true,
                            }}
                        }}
                    }}
                }}
            }});


        </script>
    </body>
    </html>
    """
    return html_content

def main():
    directory_path = os.getcwd()
    file_path = 'report/reports/data/plots/training/'
    file_path_2 = f"{file_path}val_acc_and_loss.json"
    full_file_path = os.path.join(directory_path, file_path_2)

    with open(full_file_path, "r") as json_file:
        data = json.load(json_file)

    data_x = data["x"]
    data_y1 = data["y1"]
    data_y2 = data["y2"]
    data_y3 = data["y3"]
    data_y4 = data["y4"]

    html_content = generate_html_with_chart(data_x, data_y1, data_y2, data_y3, data_y4)

    with open("report/reports/interactive_chart.html", "w") as html_file:
        html_file.write(html_content)

if __name__ == "__main__":
    main()
