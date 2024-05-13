import json


def html_accuracy_loss_chart(data_accuracy_loss, title):
    
    data_al_x            = data_accuracy_loss["x_nr"]
    data_al_accuracy     = data_accuracy_loss["accuracy"]
    data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
    data_al_loss         = data_accuracy_loss["loss"]
    data_al_val_loss     = data_accuracy_loss["val_loss"]

    html_content = f"""

    <!DOCTYPE html>
    <html>
        <head>
            <title>Training Accuracy & Loss</title>
            <title>{title}</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js"></script>
        </head>
        <body>
            
            <h2>Training</h2>
            <div class="charts_box">
                <div id="chart_accuracy" class="chart_line_1"></div>
                <div id="chart_loss" class="chart_line_1"></div>
            </div>

            <script>
                
                var js_chart_accuracy = echarts.init(document.getElementById("chart_accuracy"));
                var js_chart_loss = echarts.init(document.getElementById("chart_loss"));

                var data_al_x = {json.dumps(data_al_x)};

                var data_al_accuracy = {json.dumps(data_al_accuracy)};
                var data_al_val_accuracy = {json.dumps(data_al_val_accuracy)};

                var data_al_loss = {json.dumps(data_al_loss)};
                var data_al_val_loss = {json.dumps(data_al_val_loss)};

                var option_accuracy = {{
                    title: {{
                        text: "Accuracy"
                    }},
                    legend: {{
                        data: ["Accuracy", "Validation Accuracy"]
                    }},

                    tooltip: {{
                        trigger: "axis",
                        axisPointer: {{
                            type: "cross"
                        }}
                    }},

                    toolbox: {{
                        feature: {{
                            dataZoom: {{
                                yAxisIndex: "none"
                            }},
                            restore: {{}},
                            magicType: {{ type: ["line", "bar"] }}
                        }}
                    }},
                    grid: {{
                        left: "5%",
                        right: "5%",
                        bottom: "5%",
                        containLabel: true
                    }},
                    
                    xAxis: {{
                        type: "category",
                        boundaryGap: false,
                        data: data_al_x
                    }},
                    yAxis: {{
                        type: "value"
                    }},

                    series: [
                        {{
                            name: "Accuracy",
                            type: "line",
                            data: data_al_accuracy,
                            smooth: true,
                            markPoint: {{
                                label: {{
                                    formatter: "{{c}}"
                                }},
                                data: [
                                    {{ type: "max", name: "" }},
                                    {{ type: "min", name: "" }}
                                ]
                            }}
                        }},
                        {{
                            name: "Validation Accuracy",
                            type: "line",
                            data: data_al_val_accuracy,
                            smooth: true
                        }}
                    ],

                    dataZoom: [
                        {{
                            type: "slider"
                        }},
                        {{
                            type: "inside"
                        }}
                    ]
                }};

                var option_loss = {{
                    title: {{
                        text: "Loss"
                    }},
                    legend: {{
                        data: ["Loss", "Validation Loss"]
                    }},

                    tooltip: {{
                        trigger: "axis",
                        axisPointer: {{
                            type: "cross"
                        }}
                    }},
                    toolbox: {{
                        feature: {{
                            dataZoom: {{
                                yAxisIndex: "none"
                            }},
                            restore: {{}},
                            magicType: {{ type: ["line", "bar"] }}
                        }}
                    }},
                    grid: {{
                        left: "5%",
                        right: "5%",
                        bottom: "5%",
                        containLabel: true
                    }},

                    xAxis: {{
                        type: "category",
                        boundaryGap: false,
                        data: data_al_x
                    }},
                    yAxis: {{
                        type: "value"
                    }},

                    series: [
                        {{
                            name: "Loss",
                            type: "line",
                            data: data_al_loss,
                            smooth: true,
                            markPoint: {{
                                label: {{
                                    formatter: "{{c}}"
                                }},
                                data: [
                                    {{ type: "max", name: "y" }},
                                    {{ type: "min", name: "y" }}
                                ]
                            }}
                        }},
                        {{
                            name: "Validation Loss",
                            type: "line",
                            data: data_al_val_loss,
                            smooth: true
                        }} 
                    ],

                    dataZoom: [
                        {{
                            type: "slider"
                        }},
                        {{
                            type: "inside"
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



def html_heatmap_chart(data_heatmap):
    
    heatmap_columns = list({each_data["column"] for each_data in data_heatmap})
    heatmap_rows = list({each_data["row"] for each_data in data_heatmap})
    heatmap_rows_flip = heatmap_rows[::-1]
    heatmap_max_value = max(each_data["value"] for each_data in data_heatmap)
    
    heatmap_data_xyz = []
    for each in data_heatmap:
        column_index = heatmap_columns.index(each["column"])
        row_index = heatmap_rows_flip.index(each["row"])
        value = each["value"]
        heatmap_data_xyz.append([column_index, row_index, value])
    
    html_content =  f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Confusion Matrix</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js"></script>
        </head>
        <body>
            
            <h2>Confusion Matrix</h2>

            <div id="chart_heatmap"></div>

            <script>
                var js_chart_heatmap = echarts.init(document.getElementById("chart_heatmap"));
                var data = {json.dumps(heatmap_data_xyz)};
                var columns = {json.dumps(heatmap_columns)};
                var rows = {json.dumps(heatmap_rows_flip)};

                var option = {{
                    tooltip: {{
                        position: "top"
                    }},

                    animation: false,

                    grid: {{
                        height: "50%",
                        top: "10%"
                    }},

                    xAxis: {{
                        type: "category",
                        data: columns,
                        splitArea: {{
                            show: true
                        }}
                    }},

                    yAxis: {{
                        type: "category",
                        data: rows,
                        splitArea: {{
                            show: true
                        }}
                    }},

                    visualMap: {{
                        min: 0,
                        max: {heatmap_max_value},
                        calculable: true,
                        orient: "horizontal",
                        left: "center",
                        bottom: "15%",
                        inRange: {{
                            color: ["#e0ffff", "#006edd"]
                        }}
                    }},

                    series: [{{
                        name: "Heatmap",
                        type: "heatmap",
                        data: data,
                        label: {{
                            show: true
                        }},
                        itemStyle: {{
                            emphasis: {{
                                shadowBlur: 10,
                                shadowColor: "rgba(0, 0, 0, 0.5)"
                            }}
                        }}
                    }}],

                    dataZoom: [{{
                        type: "slider",
                        show: true,
                        xAxisIndex: [0]
                    }},
                    {{
                        type: "inside"
                    }}]

                }};
                js_chart_heatmap.setOption(option);
            </script>
        </body>
    </html>
    """

    return html_content


def html_heatmap_chart_2(data_heatmap):

    heatmap_columns = data_heatmap["columns"][:-1]
    heatmap_rows = data_heatmap["rows"]
    heatmap_values = data_heatmap["value"]

    heatmap_data = []
    for x, row in enumerate(heatmap_values):
        for y, value in enumerate(row[:-1]):
            heatmap_data.append([x, y, round(value, 3)])

    html_content =  f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Classification Report</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js"></script>
        </head>
        <body>
            
            <h2>Classification Report</h2>

            <div id="chart_heatmap"></div>

            <script>
                var js_chart_heatmap = echarts.init(document.getElementById("chart_heatmap"));
                var data = {json.dumps(heatmap_data)};
                var columns = {json.dumps(heatmap_columns)};
                var rows = {json.dumps(heatmap_rows)};

                var option = {{
                    tooltip: {{
                        position: "top"
                    }},

                    animation: false,

                    grid: {{
                        height: "50%",
                        top: "10%"
                    }},

                    xAxis: {{
                        type: "category",
                        data: rows,
                        axisLabel: {{
                            interval: 0,
                            rotate: 30,
                            fontSize: 10
                        }},
                        splitArea: {{
                            show: true
                        }}
                    }},
                    
                    yAxis: {{
                        type: "category",
                        data: columns,
                        splitArea: {{
                            show: true
                        }}
                    }},

                    visualMap: {{
                        min: 0,
                        max: 1,
                        calculable: true,
                        orient: "horizontal",
                        left: "center",
                        bottom: "15%",
                        type: "piecewise",
                        pieces: [
                            {{min: 0, max: 0.3, label: "Under 0.2", color: "#f9ff7e"}},
                            {{min: 0.2, max: 0.4, label: "0.2 - 0.4", color: "#e3ff7e"}},
                            {{min: 0.4, max: 0.6, label: "0.4 - 0.6", color: "#c4ff7e"}},
                            {{min: 0.6, max: 0.8, label: "0.6 - 0.8", color: "#85ff78"}},
                            {{min: 0.8, max: 1, label: "Over 0.8", color: "#5ed164"}}
                        ],
                    }},

                    series: [{{
                        name: "Heatmap",
                        type: "heatmap",
                        data: data,
                        label: {{
                            show: true
                        }},
                        itemStyle: {{
                            emphasis: {{
                                shadowBlur: 10,
                                shadowColor: "rgba(0, 0, 0, 0.5)"
                            }}
                        }}
                    }}],

                    dataZoom: [{{
                        type: "slider",
                        show: true,
                        xAxisIndex: [0]
                    }},
                    {{
                        type: "inside"
                    }}]
                }};
                js_chart_heatmap.setOption(option);
            </script>
        </body>
    </html>
    """

    return html_content