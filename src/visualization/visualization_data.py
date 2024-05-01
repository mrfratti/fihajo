# import json


def jason_plot_results(xlabel, history_data):
    data_info = {
        "x": xlabel,
        "y1": history_data["accuracy"],
        "y2": history_data["val_accuracy"],
        "y3": history_data["loss"],
        "y4": history_data["val_loss"]
    }

    return data_info







