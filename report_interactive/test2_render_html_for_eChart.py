import os
import json

from report_interactive.html_structure import html_start, html_accuracy_loss_chart, html_heatmap_chart, html_end

# Make function for adding +1 or build nr from jenkins too a jason file, that will be used for every stage, 
# so we can use that info to make 1 file where they can see all build graf and compare
# jason structure: sum: 5      traning: 1, 2, 3,       adv: 2, 3,       evaluate

# JASON STRUCTURE:
# build :   1, 2, 3
# adv   :   2
# evaluate  :   1, 2,3

# SUM should help with how many in totall, it help out for eks: adv where it start with 2, 3, we need a variable helper!
# USE try and catch to skip error if there is no adv file for that build nr and so on

def build_list_info(option):
    with open("build_list.json", "r") as file:
        data_build_info = json.load(file)
    
    number_last = data_build_info[option][-1]
    next_number = number_last + 1

    data_build_info[option].append(next_number)

    with open("build_list.json", "w") as file:
        json.dump(data_build_info, file, indent=4)

def main():

    main_path = "report/reports/data/plots"

    # --- HTML FOUNDATION --- |
    html_content  = html_start()


    # --- TRAINING --- |
    html_content += "<div id='content_training' class='display'><h2>TRAINING</h2>"

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
    html_content += "<div id='content_evaluate' class='display'><h2>EVALUATION</h2>"

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



    # --- ANALYZE --- |
    html_content += "<div id='content_analyze' class='display'><h2>ANALYZE</h2>"

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
    html_content += "<div id='content_aAImd' class='display'>content_aAImd</div><h2>ALL AI MODEL</h2>"


    # --- HTML FOUNDATION --- |
    html_content += html_end()

    with open("report_interactive/interactive_chart.html", "w") as html_file:
        html_file.write(html_content)



if __name__ == "__main__":
    main()
