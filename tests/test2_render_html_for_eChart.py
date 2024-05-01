import os
import json

# Make function for adding +1 or build nr from jenkins too a jason file, that will be used for every stage, 
# so we can use that info to make 1 file where they can see all build graf and compare
# jason structure: sum: 5      traning: 1, 2, 3,       adv: 2, 3,
# sum should help with how many in totall, it help out for eks: adv where it start with 2, 3, we need a variable helper!


def generate_html_with_chart(data_x, data_y1, data_y2, data_y3, data_y4):
    html_content = f"""
    <html>
    <head>

        <title>Interactive Charts</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js"></script>

        <style>
            #charts_box {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}

            .chart_line_1 {{
                width: 600px;
                height: 400px;
            }}
        </style>

    </head>
    <body>

        <div id="charts_box">
            <div id="chart_accuracy" class="chart_line_1"></div>
            <div id="chart_loss" class="chart_line_1"></div>
        </div>

        <script>

            function create_option(title_text, data_set) {{
                return {{
                    title: {{
                        text: title_text
                    }},
                    tooltip: {{
                        trigger: 'axis'
                    }},
                    legend: {{
                        data: ['Accuracy', 'Validation Accuracy', 'Loss', 'Validation Loss']
                    }},
                    grid: {{
                        left: '5%',
                        right: '5%',
                        bottom: '5%',
                        containLabel: true
                    }},
                    toolbox: {{
                        feature: {{
                            dataZoom: {{
                                yAxisIndex: 'none'
                            }},
                            restore: {{}}
                        }}
                    }},
                    xAxis: {{
                        type: 'category',
                        boundaryGap: false,
                        data: {json.dumps(data_x)}
                    }},
                    yAxis: {{
                        type: 'value'
                    }},
                    series: data_set
                }};
            }}

            var js_chart_accuracy = echarts.init(document.getElementById('chart_accuracy'));
            js_chart_accuracy.setOption(create_option('Accuracy', [
                {{
                    name: 'Accuracy',
                    type: 'line',
                    stack: 'Total',
                    data: {json.dumps(data_y1)}
                }},
                
                {{
                    name: 'Validation Accuracy',
                    type: 'line',
                    stack: 'Total',
                    data: {json.dumps(data_y2)}
                }}
            ]));

            var js_chart_loss = echarts.init(document.getElementById('chart_loss'));
            js_chart_loss.setOption(create_option('Loss', [
                {{
                    name: 'Loss',
                    type: 'line',
                    stack: 'Total',
                    data: {json.dumps(data_y3)}
                }},

                {{
                    name: 'Validation Loss',
                    type: 'line',
                    stack: 'Total',
                    data: {json.dumps(data_y4)}
                }}
            ]));

        </script>
    </body>
    </html>
    """
    return html_content

def main():

    # --- Accuracy & Loss --- |
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


    # --- Confusion Matrix --- |


    html_content = generate_html_with_chart(data_x, data_y1, data_y2, data_y3, data_y4)

    with open("report/reports/interactive_chart.html", "w") as html_file:
        html_file.write(html_content)

    


if __name__ == "__main__":
    main()
