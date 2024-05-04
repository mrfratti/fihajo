import os
import json

def html_start():
    html_content = """
    <html>
    <head>

        <title>Interactive Charts</title>
        <link rel="stylesheet" href="main.css">
        <script src="main.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js"></script>

    </head>
    <body>

    <header>
        <button onclick="option_show('content_training')">Training</button>
        <button onclick="option_show('content_evaluate')">Evaluate</button>
        <button onclick="option_show('content_analyze')">Analyze</button>
        <button onclick="option_show()">Adversarial Training</button>
        <button onclick="option_show()">All AI model data</button>
    </header>
    
    """

    return html_content



def html_accuracy_loss_chart(data_x, accuracy, val_accuracy, loss, val_loss):
    html_content = f"""
    
    <div class="charts_box">
        <h2>Training</h2>
        <div id="chart_accuracy" class="chart_line_1"></div>
        <div id="chart_loss" class="chart_line_1"></div>
    </div>

    <script>
        
        var js_chart_accuracy = echarts.init(document.getElementById('chart_accuracy'));
        var js_chart_loss = echarts.init(document.getElementById('chart_loss'));

        var option_accuracy = {{
            title: {{
                text: 'Accuracy'
            }},
            legend: {{
                data: ['Accuracy', 'Validation Accuracy']
            }},

            tooltip: {{
                trigger: 'axis',
                axisPointer: {{
                    type: 'cross'
                }}
            }},
            toolbox: {{
                feature: {{
                    dataZoom: {{
                        yAxisIndex: 'none'
                    }},
                    restore: {{}},
                    magicType: {{ type: ['line', 'bar'] }}
                }}
            }},
            grid: {{
                left: '5%',
                right: '5%',
                bottom: '5%',
                containLabel: true
            }},
            
            xAxis: {{
                type: 'category',
                boundaryGap: false,
                data: {json.dumps(data_x)}
            }},
            yAxis: {{
                type: 'value'
            }},

            series: [
                {{
                    name: 'Accuracy',
                    type: 'line',
                    data: {json.dumps(accuracy)},
                    smooth: true,
                    markPoint: {{
                        label: {{
                            formatter: function (param) {{
                                return param.name + ': ' + param.value.toFixed(2);
                            }}
                        }},
                        data: [
                            {{ type: 'max', name: 'Max' }},
                            {{ type: 'min', name: 'Min' }}
                        ]
                    }}
                }},
                {{
                    name: 'Validation Accuracy',
                    type: 'line',
                    data: {json.dumps(val_accuracy)},
                    smooth: true
                }}
            ]
        }};

        var option_loss = {{
            title: {{
                text: 'Loss'
            }},
            legend: {{
                data: ['Loss', 'Validation Loss']
            }},

            tooltip: {{
                trigger: 'axis',
                axisPointer: {{
                    type: 'cross'
                }}
            }},
            toolbox: {{
                feature: {{
                    dataZoom: {{
                        yAxisIndex: 'none'
                    }},
                    restore: {{}},
                    magicType: {{ type: ['line', 'bar'] }}
                }}
            }},
            grid: {{
                left: '5%',
                right: '5%',
                bottom: '5%',
                containLabel: true
            }},

            xAxis: {{
                type: 'category',
                boundaryGap: false,
                data: {json.dumps(data_x)}
            }},
            yAxis: {{
                type: 'value'
            }},

            series: [
                {{
                    name: 'Loss',
                    type: 'line',
                    data: {json.dumps(loss)},
                    smooth: true,
                    markPoint: {{
                        label: {{
                            formatter: function (param) {{
                                return param.name + ': ' + param.value.toFixed(2);
                            }}
                        }},
                        data: [
                            {{ type: 'max', name: 'Max' }},
                            {{ type: 'min', name: 'Min' }}
                        ]
                    }}
                }},
                {{
                    name: 'Validation Loss',
                    type: 'line',
                    data: {json.dumps(val_loss)},
                    smooth: true
                }}
            ]
        }};

        js_chart_accuracy.setOption(option_accuracy);
        js_chart_loss.setOption(option_loss);

    </script>
 
    """
    return html_content



def html_heatmap_chart(data_heatmap, heatmap_columns, heatmap_rows, heatmap_max_value):
    data_value = json.dumps(data_heatmap)
    data_columns = json.dumps(heatmap_columns)
    data_rows = json.dumps(heatmap_rows)

    html_content =  f"""

        <div id="chart_heatmap" class="display"></div>

        <script>
            var js_chart_heatmap = echarts.init(document.getElementById('chart_heatmap'));
            var data = {data_value};
            var columns = {data_columns};
            var rows = {data_rows};

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
                    type: 'category',
                    data: columns,
                    splitArea: {{
                        show: true
                    }}
                }},
                yAxis: {{
                    type: 'category',
                    data: rows,
                    splitArea: {{
                        show: true
                    }}
                }},

                visualMap: {{
                    min: 0,
                    max: {heatmap_max_value},
                    calculable: true,
                    orient: 'horizontal',
                    left: 'center',
                    bottom: '15%'
                }},

                series: [{{
                    name: 'Heatmap',
                    type: 'heatmap',
                    data: data.map(function (item) {{
                        return [columns.indexOf(item.column), rows.indexOf(item.row), item.value];
                    }}),
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



# Make function for adding +1 or build nr from jenkins too a jason file, that will be used for every stage, 
# so we can use that info to make 1 file where they can see all build graf and compare
# jason structure: sum: 5      traning: 1, 2, 3,       adv: 2, 3,       evaluate

# JASON STRUCTURE:
# build :   1,2,3
# adv   :   2
# evaluate  :   1,2,3

# SUM should help with how many in totall, it help out for eks: adv where it start with 2, 3, we need a variable helper!
# USE try and catch to skip error if there is no adv file for that build nr and so on

def main():

    main_path = "report/reports/data/plots"

    # --- HTML FOUNDATION --- |
    html_content  = html_start()


    # --- TRAINING --- |
    html_content += "<div id='content_training' class='display'>"

    # --- Accuracy & Loss --- |
    
    file_path = main_path + "/training"
    full_file_path = os.path.join(os.getcwd(), f"{file_path}/val_acc_and_loss.json")

    with open(full_file_path, "r") as json_file:
        data_accuracy_loss = json.load(json_file)

    data_al_x = data_accuracy_loss["x"]
    data_al_accuracy = data_accuracy_loss["accuracy"]
    data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
    data_al_loss = data_accuracy_loss["loss"]
    data_al_val_loss = data_accuracy_loss["val_loss"]

    html_content += html_accuracy_loss_chart(data_al_x, data_al_accuracy, data_al_val_accuracy, data_al_loss, data_al_val_loss)

    html_content += "</div>"



    # --- EVALUATION --- |
    html_content += "<div id='content_evaluate' class='display'>"

    # --- Confusion Matrix --- |

    file_path = main_path + "/evaluation"
    full_file_path = os.path.join(os.getcwd(), f"{file_path}/confusion_matrix.json")

    with open(full_file_path, "r") as json_file:
        data_heatmap = json.load(json_file)

    heatmap_columns = list({item['column'] for item in data_heatmap})
    heatmap_rows = list({item['row'] for item in data_heatmap})
    heatmap_max_value = max(item['value'] for item in data_heatmap)

    html_content += html_heatmap_chart(data_heatmap, heatmap_columns, heatmap_rows, heatmap_max_value)

    
    # --- Classification Report --- |

    file_path = main_path + "/evaluation"
    full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_classification_report.html")
    with open(full_file_path, "r") as file:
        data_content = file.readlines()

    content_start = data_content.index("<body>\n") + 1
    content_end = data_content.index("</body>\n")
    content_content = data_content[content_start:content_end]
    content_string = "".join(content_content)
    html_content += content_string


    html_content += "</div>"



    # --- ANALYYZE --- |
    html_content += "<div id='content_analyze' class='display'>"

    # --- PCS & Mean Softmax Score --- |
    file_path = main_path + "/analyze"
    full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_pcs_mean_softmax.html")
    with open(full_file_path, "r") as file:
        data_content = file.readlines()

    content_start = data_content.index("<body>\n") + 1
    content_end = data_content.index("</body>\n")
    content_content = data_content[content_start:content_end]
    content_string = "".join(content_content)
    html_content += content_string


    # --- Entropy Scores --- |

    file_path = main_path + "/analyze"
    full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_dist_entropy_scores.html")
    with open(full_file_path, "r") as file:
        data_content = file.readlines()

    content_start = data_content.index("<body>\n") + 1
    content_end = data_content.index("</body>\n")
    content_content = data_content[content_start:content_end]
    content_string = "".join(content_content)
    html_content += content_string


    html_content += "</div>"



    # --- ALL AI MODEL --- |

    # Show pie chart of all ai model accurcy, so they know what to chooce
    html_content += "<div id='content_aAImd' class='display'>content_aAImd</div>"


    # --- HTML FOUNDATION --- |
    html_content += html_end()

    with open("report_interactive/interactive_chart.html", "w") as html_file:
        html_file.write(html_content)



if __name__ == "__main__":
    main()
