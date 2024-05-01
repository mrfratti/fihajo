import os
import json

# Make function for adding +1 or build nr from jenkins too a jason file, that will be used for every stage, 
# so we can use that info to make 1 file where they can see all build graf and compare
# jason structure: sum: 5      traning: 1, 2, 3,       adv: 2, 3,
# sum should help with how many in totall, it help out for eks: adv where it start with 2, 3, we need a variable helper!


def html_start():
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

            .chart_line_2 {{
                width: 600px;
                height: 400px;
            }}

            #heatmap {{
                width: 800px;
                height: 600px;
                margin: 50px auto;
            }}
        </style>

    </head>
    <body>
    """

    return html_content



def html_accuracy_loss_chart(data_x, data_y1, data_y2, data_y3, data_y4):
    html_content = f"""
    <div id="charts_box">
        <h>Accuracy & Loss<h>
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
    """
    return html_content



def html_heatmap_chart(data):
    json_data = json.dumps(data)
    value_max = max(item['value'] for item in data)

    html_content = f"""

    <div id="charts_box">
        <h1>Heatmap</h1>
        <div id="chart_heatmap" class="chart_line_2"></div>
    </div>

    <script>
        var js_chart_heatmap = echarts.init(document.getElementById('chart_heatmap'));
        var data = {json_data};

        var option = {{
            tooltip: {{
                position: 'top',
                formatter: function (params) {{
                    return 'Value: ' + params.value[2];
                }}
            }},

            animation: false,
            grid: {{
                height: '50%',
                top: '10%'
            }},

            xAxis: {{
                splitArea: {{
                    show: true
                }}
                type: 'category',
                data: data.map(item => item.column_y.toString()),
            }},
            yAxis: {{
                splitArea: {{
                    show: true
                }}
                type: 'category',
                data: data.map(item => item.row_x.toString()),
            }},
            
            visualMap: {{
                min: 0,
                max: {value_max},
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '15%'
            }},

            series: [{{
                name: 'Heatmap',
                type: 'heatmap',
                data: data.map(item => [item.column_y, item.row_x, item.value]),
                label: {{
                    show: true
                }},
                itemStyle: {{
                    emphasis: {{
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }}
                }}
            }}]
        }};

        js_chart_heatmap.setOption(option);
    </script>
    """

    return html_content




def html_end():
    html_content = f"""
    # </body>
    # </html>
    """
    return html_content




def main():

    # --- Accuracy & Loss --- |
    file_path = 'report/reports/data/plots/training'
    file_path_2 = f"{file_path}/val_acc_and_loss.json"
    full_file_path = os.path.join(os.getcwd(), file_path_2)

    with open(full_file_path, "r") as json_file:
        data_accuracy_loss = json.load(json_file)

    data_al_x = data_accuracy_loss["x"]
    data_al_y1 = data_accuracy_loss["y1"]
    data_al_y2 = data_accuracy_loss["y2"]
    data_al_y3 = data_accuracy_loss["y3"]
    data_al_y4 = data_accuracy_loss["y4"]


    # --- Confusion Matrix --- |
    file_path = 'report/reports/data/plots/evaluation'
    file_path_2 = f"{file_path}/confusion_matrix.json"
    full_file_path = os.path.join(os.getcwd(), file_path_2)

    with open(full_file_path, "r") as json_file:
        data_heatmap = json.load(json_file)
        

    # --- HTML FOUNDATION --- |
    html_content = html_start()
    html_content += html_accuracy_loss_chart(data_al_x, data_al_y1, data_al_y2, data_al_y3, data_al_y4)
    html_content += html_heatmap_chart(data_heatmap)
    html_content += html_end()

    with open("report/reports/interactive_chart.html", "w") as html_file:
        html_file.write(html_content)

    


if __name__ == "__main__":
    main()
