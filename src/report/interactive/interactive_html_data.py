import json


def html_accuracy_loss_chart(data_accuracy_loss):
    
    data_al_x = data_accuracy_loss["x"]
    data_al_accuracy = data_accuracy_loss["accuracy"]
    data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
    data_al_loss = data_accuracy_loss["loss"]
    data_al_val_loss = data_accuracy_loss["val_loss"]

    html_content = f"""

    <!DOCTYPE html>
    <html>
        <head>
            <title>Training Accuracy & Loss</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js"></script>
            <style>
                body {{
                    background-color: #d5d5d5;
                }}
                .charts_box {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 50px auto;
                }}
                .chart_line_1 {{
                    width: 600px;
                    height: 400px;
                }}
            </style>
        </head>
        <body>
            
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
                        data: {json.dumps(data_al_x)}
                    }},
                    yAxis: {{
                        type: 'value'
                    }},

                    series: [
                        {{
                            name: 'Accuracy',
                            type: 'line',
                            data: {json.dumps(data_al_accuracy)},
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
                            data: {json.dumps(data_al_val_accuracy)},
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
                        data: {json.dumps(data_al_x)}
                    }},
                    yAxis: {{
                        type: 'value'
                    }},

                    series: [
                        {{
                            name: 'Loss',
                            type: 'line',
                            data: {json.dumps(data_al_loss)},
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
                            data: {json.dumps(data_al_val_loss)},
                            smooth: true
                        }}
                    ]
                }};

                js_chart_accuracy.setOption(option_accuracy);
                js_chart_loss.setOption(option_loss);

            </script>
        </body>
    </html>
    """
    return html_content



def html_heatmap_chart(data_heatmap, heatmap_columns, heatmap_rows, heatmap_max_value):
    data_value = json.dumps(data_heatmap)
    data_columns = json.dumps(heatmap_columns)
    data_rows = json.dumps(heatmap_rows)

    html_content =  f"""

        <div id="chart_heatmap"></div>

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
