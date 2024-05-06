import json
import os
import datetime


class Interactive_Html_Data:
    def __init__(self):
        self._main_path = "src/report/reports/data/plots"
        # self._build_nr = self.build_nr_now("build_nr")

    def html_start(self):
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



    def html_accuracy_loss_chart(self, data_x, accuracy, val_accuracy, loss, val_loss):
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



    def html_heatmap_chart(self, data_heatmap, heatmap_columns, heatmap_rows, heatmap_max_value):
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




    def html_end(self):
        html_content = """

        # </body>
        # </html>
        """
        return html_content
    






    # def generate(self):
    #     html_content = self.html_start()
    #     html_content = self.html_training(html_content)
    #     html_content = self.html_evaluation(html_content)
    #     html_content = self.html_analyse(html_content)
    #     html_content = self.html_adv(html_content)

    #     # --- HTML END --- |
    #     html_content += Interactive_Html_Data().html_end()

    #     with open("src/report_interactive/interactive_chart.html", "w") as html_file:
    #         html_file.write(html_content)

    #     self.delete_cheack_file()
    



    # def html_start(self):
    #     # --- HTML FOUNDATION --- |
    #     html_content  = Interactive_Html_Data().html_start()

    #     if self._build_nr == "ERROR no file":
    #         html_content += "<h1> !NEED TO RUN AI MODEL WITH COMMAND! <h1>"

    #     return html_content 
    

    # def html_training(self, html_content):
    #     # --- TRAINING --- |
    #     file_path = self._main_path + "/training"
    #     html_content += "<div id='content_training' class='display'><h2>TRAINING</h2><br><br>"

    #     # --- Accuracy & Loss --- |
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/val_acc_and_loss{self._build_nr}.json")

    #     if os.path.exists(full_file_path):
    #         with open(full_file_path, "r") as json_file:
    #             data_accuracy_loss = json.load(json_file)

    #         data_al_x = data_accuracy_loss["x"]
    #         data_al_accuracy = data_accuracy_loss["accuracy"]
    #         data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
    #         data_al_loss = data_accuracy_loss["loss"]
    #         data_al_val_loss = data_accuracy_loss["val_loss"]

    #         html_content += "<p>" + f"{file_path}/val_acc_and_loss{self._build_nr}.json" + "<p>"
    #         html_content += self._html_data.html_accuracy_loss_chart(data_al_x, data_al_accuracy, data_al_val_accuracy, data_al_loss, data_al_val_loss)

    #     # --- |
    #     html_content += "</div>"

    #     return html_content
    

    # def html_evaluation(self, html_content):
    #     # --- EVALUATION --- |
    #     file_path = self._main_path + "/evaluation"
    #     html_content += "<div id='content_evaluate' class='display'><h2>EVALUATION</h2>"

    #     # --- Confusion Matrix --- |
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/confusion_matrix{self._build_nr}.json")
    #     if os.path.exists(full_file_path):
    #         with open(full_file_path, "r") as json_file:
    #             data_heatmap = json.load(json_file)

    #         heatmap_columns = list({item['column'] for item in data_heatmap})
    #         heatmap_rows = list({item['row'] for item in data_heatmap})
    #         heatmap_max_value = max(item['value'] for item in data_heatmap)

    #         html_content += "<p>" + f"{file_path}/confusion_matrix{self._build_nr}.json" + "<p>"
    #         html_content += self._html_data.html_heatmap_chart(data_heatmap, heatmap_columns, heatmap_rows, heatmap_max_value)

    #     # --- Classification Report --- |
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_classification_report{self._build_nr}.html")
    #     html_content += self.create_div_file_html(full_file_path)
    #     html_content += "<p>" + f"{file_path}/plot_classification_report{self._build_nr}.html" + "<p>"

    #     # --- plot_predictions --- |
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_predictions{self._build_nr}.png")
    #     html_content += f"<img src={full_file_path} alt='plot predictions'>"
    #     html_content += "<p>" + f"{file_path}/plot_predictions{self._build_nr}.html" + "<p>"

    #     # --- |
    #     html_content += "</div>"

    #     return html_content



    # def html_analyse(self, html_content):
    # # --- ANALYZE --- |
    #     file_path = self._main_path + "/analyze"
    #     html_content += "<div id='content_analyze' class='display'><h2>ANALYZE</h2>"

    #     # --- PCS & Mean Softmax Score --- |
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_pcs_mean_softmax{self._build_nr}.html")
    #     html_content += self.create_div_file_html(full_file_path)
    #     html_content += "<p>" + f"{file_path}/plot_pcs_mean_softmax{self._build_nr}.html" + "<p>"

    #     # --- Entropy Scores --- |
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_dist_entropy_scores{self._build_nr}.html")
    #     html_content += self.create_div_file_html(full_file_path)
    #     html_content += "<p>" + f"{file_path}/plot_dist_entropy_scores{self._build_nr}.html" + "<p>"

    #     # --- Entropy Scatter Scores --- |

    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_predictive_conf_entropy_scores{self._build_nr}.html")
    #     html_content += self.create_div_file_html(full_file_path)
    #     html_content += "<p>" + f"{file_path}/plot_predictive_conf_entropy_scores{self._build_nr}.html" + "<p>"
        
    #     # --- |
    #     html_content += "</div>"

    #     return html_content


    # def html_adv(self, html_content):
    #     # --- ADV --- |
    #     html_content += "<div id='content_analyze' class='display'><h2>Adversarial Training</h2>"

    #     # --- plot_adversarial_training_results --- |
    #     file_path = self._main_path + "/training"
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_adversarial_training_results{self._build_nr}.json")

    #     if os.path.exists(full_file_path):
    #         with open(full_file_path, "r") as json_file:
    #             data_accuracy_loss = json.load(json_file)

    #         data_al_x = data_accuracy_loss["x"]
    #         data_al_accuracy = data_accuracy_loss["accuracy"]
    #         data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
    #         data_al_loss = data_accuracy_loss["loss"]
    #         data_al_val_loss = data_accuracy_loss["val_loss"]

    #         html_content += "<p>" + f"{file_path}/plot_adversarial_training_results{self._build_nr}.json" + "<p>"
    #         html_content += Interactive_Html_Data().html_accuracy_loss_chart(data_al_x, data_al_accuracy, data_al_val_accuracy, data_al_loss, data_al_val_loss)

    #     html_content += "</div>"

    #     # need to add try and catch for to find if the file exist

    #     file_path = self._main_path + "/evaluation"
    #     full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_accuracy_comparison{self._build_nr}.html")
    #     html_content += self.create_div_file_html(full_file_path)
    #     html_content += "<p>" + f"{file_path}/plot_accuracy_comparison{self._build_nr}.html" + "<p>"

    #     return html_content


    # def create_cheack_file(self):

    #     date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    #     default_data = {
    #         "build_nr": date_time,
    #     }
        
    #     file_path = "src/report_interactive/build_list.json"
    #     if not os.path.exists(file_path):
    #         with open(file_path, 'w') as file:
    #             json.dump(default_data, file, indent=4)

    # def delete_cheack_file(self):
    #     file_path = "src/report_interactive/build_list.json"
    #     if os.path.exists(file_path):
    #         os.remove(file_path)



    # def build_nr_now(self, option):
    #     file_path = "src/report_interactive/build_list.json"
    #     full_file_path = os.path.join(os.getcwd(), file_path)

    #     if not os.path.exists(full_file_path):
    #         with open(full_file_path, "r") as file:
    #             data_build_info = json.load(file)
            
    #         number_last = data_build_info[option]
    #         number_last_text = "_build_" + number_last

    #         return number_last_text
        
    #     # else:
    #     #     return "ERROR no file"


    # def create_div_file_html(self, full_file_path):
    #     html_content = ""
    #     if os.path.exists(full_file_path):
    #         with open(full_file_path, "r") as file:
    #             data_content = file.readlines()

    #         content_start = data_content.index("<body>\n") + 1
    #         content_end = data_content.index("</body>\n")
    #         content_content = data_content[content_start:content_end]
    #         content_string = "".join(content_content)
    #         html_content += content_string
    #     return html_content