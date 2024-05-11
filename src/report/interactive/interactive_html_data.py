import json


def html_heatmap_chart(data_heatmap):

    heatmap_columns = list({each_data["column"] for each_data in data_heatmap})
    heatmap_rows = list({each_data["row"] for each_data in data_heatmap})
    heatmap_rows = heatmap_rows[::-1]
    heatmap_max_value = max(each_data["value"] for each_data in data_heatmap)
    
    html_content =  f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Confusion Matrix</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js"></script>
            <style>
                body {{
                    background-color: #f2f2d2;
                }}
                #chart_heatmap {{
                    width: 800px;
                    height: 600px;
                    margin: 50px auto;
                }}
            </style>
        </head>
        <body>
            
            <h2>Confusion Matrix</h2>

            <div id="chart_heatmap"></div>

            <script>
                var js_chart_heatmap = echarts.init(document.getElementById('chart_heatmap'));
                var data = {json.dumps(data_heatmap)};
                var columns = {json.dumps(heatmap_columns)};
                var rows = {json.dumps(heatmap_rows)};

                var option = {{
                    tooltip: {{
                        position: "top",
                        formatter: function (params) {{
                            return "Value: " + params.value[2];
                        }}
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
                        bottom: "15%"
                    }},

                    series: [{{
                        name: "Heatmap",
                        type: "heatmap",
                        data: data.map(function (item) {{
                            return [columns.indexOf(item.column), rows.indexOf(item.row), item.value];
                        }}),
                        label: {{
                            show: true
                        }},
                        itemStyle: {{
                            emphasis: {{
                                shadowBlur: 10,
                                shadowColor: "rgba(0, 0, 0, 0.5)"
                            }}
                        }}
                    }}]
                }};
                js_chart_heatmap.setOption(option);
            </script>
        </body>
    </html>
    """

    return html_content



