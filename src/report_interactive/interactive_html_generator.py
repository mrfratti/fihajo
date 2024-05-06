import os
import json
import datetime

from src.report_interactive.interactive_html_data import Interactive_Html_Data


class Interactive_Html_Generator:
    def __init__(self):
        self._main_path = "src/report/reports/data/plots"
        self._build_nr = self.build_nr_now("build_nr")

    def generate(self):
        html_content = self.html_start()
        html_content = self.html_training(html_content)
        html_content = self.html_evaluation(html_content)
        html_content = self.html_analyse(html_content)
        html_content = self.html_adv(html_content)

        # --- HTML END --- |
        html_content += Interactive_Html_Data().html_end()

        with open("src/report_interactive/interactive_chart.html", "w") as html_file:
            html_file.write(html_content)

        self.delete_cheack_file()
    



    def html_start(self):
        # --- HTML FOUNDATION --- |
        html_content  = Interactive_Html_Data().html_start()

        if self._build_nr == "ERROR no file":
            html_content += "<h1> !NEED TO RUN AI MODEL WITH COMMAND! <h1>"

        return html_content 
    

    def html_training(self, html_content):
        # --- TRAINING --- |
        file_path = self._main_path + "/training"
        html_content += "<div id='content_training' class='display'><h2>TRAINING</h2><br><br>"

        # --- Accuracy & Loss --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/val_acc_and_loss{self._build_nr}.json")

        if os.path.exists(full_file_path):
            with open(full_file_path, "r") as json_file:
                data_accuracy_loss = json.load(json_file)

            data_al_x = data_accuracy_loss["x"]
            data_al_accuracy = data_accuracy_loss["accuracy"]
            data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
            data_al_loss = data_accuracy_loss["loss"]
            data_al_val_loss = data_accuracy_loss["val_loss"]

            html_content += "<p>" + f"{file_path}/val_acc_and_loss{self._build_nr}.json" + "<p>"
            html_content += self._html_data.html_accuracy_loss_chart(data_al_x, data_al_accuracy, data_al_val_accuracy, data_al_loss, data_al_val_loss)

        # --- |
        html_content += "</div>"

        return html_content
    

    def html_evaluation(self, html_content):
        # --- EVALUATION --- |
        file_path = self._main_path + "/evaluation"
        html_content += "<div id='content_evaluate' class='display'><h2>EVALUATION</h2>"

        # --- Confusion Matrix --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/confusion_matrix{self._build_nr}.json")
        if os.path.exists(full_file_path):
            with open(full_file_path, "r") as json_file:
                data_heatmap = json.load(json_file)

            heatmap_columns = list({item['column'] for item in data_heatmap})
            heatmap_rows = list({item['row'] for item in data_heatmap})
            heatmap_max_value = max(item['value'] for item in data_heatmap)

            html_content += "<p>" + f"{file_path}/confusion_matrix{self._build_nr}.json" + "<p>"
            html_content += self._html_data.html_heatmap_chart(data_heatmap, heatmap_columns, heatmap_rows, heatmap_max_value)

        # --- Classification Report --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_classification_report{self._build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<p>" + f"{file_path}/plot_classification_report{self._build_nr}.html" + "<p>"

        # --- plot_predictions --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_predictions{self._build_nr}.png")
        html_content += f"<img src={full_file_path} alt='plot predictions'>"
        html_content += "<p>" + f"{file_path}/plot_predictions{self._build_nr}.html" + "<p>"

        # --- |
        html_content += "</div>"

        return html_content



    def html_analyse(self, html_content):
    # --- ANALYZE --- |
        file_path = self._main_path + "/analyze"
        html_content += "<div id='content_analyze' class='display'><h2>ANALYZE</h2>"

        # --- PCS & Mean Softmax Score --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_pcs_mean_softmax{self._build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<p>" + f"{file_path}/plot_pcs_mean_softmax{self._build_nr}.html" + "<p>"

        # --- Entropy Scores --- |
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_dist_entropy_scores{self._build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<p>" + f"{file_path}/plot_dist_entropy_scores{self._build_nr}.html" + "<p>"

        # --- Entropy Scatter Scores --- |

        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_predictive_conf_entropy_scores{self._build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<p>" + f"{file_path}/plot_predictive_conf_entropy_scores{self._build_nr}.html" + "<p>"
        
        # --- |
        html_content += "</div>"

        return html_content


    def html_adv(self, html_content):
        # --- ADV --- |
        html_content += "<div id='content_analyze' class='display'><h2>Adversarial Training</h2>"

        # --- plot_adversarial_training_results --- |
        file_path = self._main_path + "/training"
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_adversarial_training_results{self._build_nr}.json")

        if os.path.exists(full_file_path):
            with open(full_file_path, "r") as json_file:
                data_accuracy_loss = json.load(json_file)

            data_al_x = data_accuracy_loss["x"]
            data_al_accuracy = data_accuracy_loss["accuracy"]
            data_al_val_accuracy = data_accuracy_loss["val_accuracy"]
            data_al_loss = data_accuracy_loss["loss"]
            data_al_val_loss = data_accuracy_loss["val_loss"]

            html_content += "<p>" + f"{file_path}/plot_adversarial_training_results{self._build_nr}.json" + "<p>"
            html_content += Interactive_Html_Data().html_accuracy_loss_chart(data_al_x, data_al_accuracy, data_al_val_accuracy, data_al_loss, data_al_val_loss)

        html_content += "</div>"

        # need to add try and catch for to find if the file exist

        file_path = self._main_path + "/evaluation"
        full_file_path = os.path.join(os.getcwd(), f"{file_path}/plot_accuracy_comparison{self._build_nr}.html")
        html_content += self.create_div_file_html(full_file_path)
        html_content += "<p>" + f"{file_path}/plot_accuracy_comparison{self._build_nr}.html" + "<p>"

        return html_content


    def create_cheack_file(self):

        date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        default_data = {
            "build_nr": date_time,
        }
        
        file_path = "src/report_interactive/build_list.json"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump(default_data, file, indent=4)

    def delete_cheack_file(self):
        file_path = "src/report_interactive/build_list.json"
        if os.path.exists(file_path):
            os.remove(file_path)



    def build_nr_now(self, option):
        file_path = "src/report_interactive/build_list.json"
        full_file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(full_file_path):
            with open(full_file_path, "r") as file:
                data_build_info = json.load(file)
            
            number_last = data_build_info[option]
            number_last_text = "_build_" + number_last

            return number_last_text
        
        # else:
        #     return "ERROR no file"


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
