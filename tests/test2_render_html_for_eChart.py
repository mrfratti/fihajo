import os
import json

def generate_html_with_chart(data_x, data_y1, data_y2, data_y3, data_y4):
    html_content = f"""
    <html>
    <head>

        <title>Interactive Line Chart</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            canvas {{
                display: block;
                margin: 0 auto;
            }}
        </style>

    </head>

    <body>
        <canvas id="chart_1" width="600" height="400"></canvas>

        <script>
            var ctx = document.getElementById('chart_1').getContext('2d');
            var chart_1 = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(data_x)},
                    datasets: [
                        {{
                            label: 'Accuracy',
                            data: {json.dumps(data_y1)},
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0
                        }},
                        {{
                            label: 'Validation Accuracy',
                            data: {json.dumps(data_y2)},
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0
                        }},
                        {{
                            label: 'Loss',
                            data: {json.dumps(data_y3)},
                            borderColor: 'rgba(255, 206, 86, 1)',
                            backgroundColor: 'rgba(255, 206, 86, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0
                        }},
                        {{
                            label: 'Validation Loss',
                            data: {json.dumps(data_y4)},
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0
                        }}
                    ]
                }},

                options: {{
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }},
                    tooltips: {{
                        mode: 'index',
                        intersect: false
                    }},
                    hover: {{
                        mode: 'nearest',
                        intersect: true
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

    with open("interactive_line_chart.html", "w") as html_file:
        html_file.write(html_content)

if __name__ == "__main__":
    main()
