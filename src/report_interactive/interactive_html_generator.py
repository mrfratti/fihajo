import os
import json

from src.report_interactive.interactive_html_data import Interactive_Html_Data


class Interactive_Html_Generator:
    def __init__(self):
        self._html_data = Interactive_Html_Data()

    def generate(self):

        main_path = "src/report/reports/data/plots"
        

        # --- HTML FOUNDATION --- |
        html_content  = self._html_data.html_start()
        build_nr = self.build_nr_now("build_nr")


        # --- TRAINING --- |
        file_path = main_path + "/training"
        html_content += "<div id='content_training' class='display'><h2>TRAINING</h2>"

        # --- Accuracy & Loss --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/val_acc_and_loss{build_nr}.json")
        html_content += "<h3>" + f"{file_path}/val_acc_and_loss{build_nr}.json" + "<h3>"

        if os.path.exists(full_file_path):
            with open(full_file_path, "r") as json_file:
                data_accuracy_loss = json.load(json_file)

            data_al_x = data_accuracy_loss["x"]
            data_al_accuracy = data_accuracy_loss["accuracy"]
            data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
            data_al_loss = data_accuracy_loss["loss"]
            data_al_val_loss = data_accuracy_loss["val_loss"]

            html_content += self._html_data.html_accuracy_loss_chart(data_al_x, data_al_accuracy, data_al_val_accuracy, data_al_loss, data_al_val_loss)

        # --- |
        html_content += "</div>"


        # --- EVALUATION --- |
        file_path = main_path + "/evaluation"
        html_content += "<div id='content_evaluate' class='display'><h2>EVALUATION</h2>"

        # --- Confusion Matrix --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/confusion_matrix{build_nr}.json")
        html_content += "<h3>" + f"{file_path}/confusion_matrix{build_nr}.json" + "<h3>"
        if os.path.exists(full_file_path):
            with open(full_file_path, "r") as json_file:
                data_heatmap = json.load(json_file)

            heatmap_columns = list({item['column'] for item in data_heatmap})
            heatmap_rows = list({item['row'] for item in data_heatmap})
            heatmap_max_value = max(item['value'] for item in data_heatmap)

            html_content += self._html_data.html_heatmap_chart(data_heatmap, heatmap_columns, heatmap_rows, heatmap_max_value)

        # --- Classification Report --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_classification_report{build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<h3>" + f"{file_path}/plot_classification_report{build_nr}.html" + "<h3>"

        # --- |
        html_content += "</div>"


        # --- ANALYZE --- |
        file_path = main_path + "/analyze"
        html_content += "<div id='content_analyze' class='display'><h2>ANALYZE</h2>"

        # --- PCS & Mean Softmax Score --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_pcs_mean_softmax{build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<h3>" + f"{file_path}/plot_pcs_mean_softmax{build_nr}.html" + "<h3>"

        # --- Entropy Scores --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_dist_entropy_scores{build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<h3>" + f"{file_path}/plot_dist_entropy_scores{build_nr}.html" + "<h3>"

        # --- Entropy Scatter Scores --- |

        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_predictive_conf_entropy_scores{build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<h3>" + f"{file_path}/plot_predictive_conf_entropy_scores{build_nr}.html" + "<h3>"
        
        # --- |
        html_content += "</div>"


        # --- ADV --- |
        html_content += "<div id='content_analyze' class='display'><h2>Adversarial Training</h2>"

        # --- plot_adversarial_training_results --- |
        file_path = main_path + "/training" # from path: training
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_adversarial_training_results{build_nr}.json")
        html_content += "<h3>" + f"{file_path}/plot_adversarial_training_results{build_nr}.json" + "<h3>"

        if os.path.exists(full_file_path):
            with open(full_file_path, "r") as json_file:
                data_accuracy_loss = json.load(json_file)

            data_al_x = data_accuracy_loss["x"]
            data_al_accuracy = data_accuracy_loss["accuracy"]
            data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
            data_al_loss = data_accuracy_loss["loss"]
            data_al_val_loss = data_accuracy_loss["val_loss"]

            html_content += self._html_data.html_accuracy_loss_chart(data_al_x, data_al_accuracy, data_al_val_accuracy, data_al_loss, data_al_val_loss)

        html_content += "</div>"

        # need to add try and catch for to find if the file exist

        # --- ALL AI MODEL --- |
  
        html_content += "<div id='content_analyze' class='display'><h2>ALL AI MODEL</h2>"

        # Show pie chart of all ai model accurcy, so they know what to chooce
        html_content += "</div>"


        # --- HTML FOUNDATION --- |
        html_content += self._html_data.html_end()

        with open("src/report_interactive/interactive_chart.html", "w") as html_file:
            html_file.write(html_content)

        self.build_list_info("build_nr")
    

    def create_cheack_file(self):
        default_data = {
            "build_nr": [0],
            "training": [0],
            "evaluation": [0],
            "analyze": [0],
            "adversarial_training": [0],
            "adversarial_evaluation": [0]
        }

        file_path = "report_interactive/build_list.json"

        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump(default_data, file, indent=4)




    def build_nr_now(self, option):
        file_path = "report_interactive/build_list.json"
        full_file_path = os.path.join(os.getcwd(), file_path)

        with open(full_file_path, "r") as file:
            data_build_info = json.load(file)

        number_last = data_build_info[option][-1]
        number_last_text = "_build_" + str(number_last)

        return number_last_text


    def build_list_info(self, option):
        file_path = "report_interactive/build_list.json"
        full_file_path = os.path.join(os.getcwd(), file_path)

        with open(full_file_path, "r") as file:
            data_build_info = json.load(file)
        
        number_last = data_build_info["build_nr"][-1]
        number_next = number_last + 1

        data_build_info[option].append(number_next)

        with open(full_file_path, "w") as file:
            json.dump(data_build_info, file, indent=4)


    def create_div_file_html(self, full_file_path):
        html_content = ""
        if os.path.exists(full_file_path):
            with open(full_file_path, "r") as file:
                data_content = file.readlines()

            content_start = data_content.index("<body>\n") + 1
            content_end = data_content.index("</body>\n")
            content_content = data_content[content_start:content_end]
            content_string = "".join(content_content)
            html_content += content_string
        return html_content
