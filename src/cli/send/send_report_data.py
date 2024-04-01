import json
import logging
import os

from matplotlib.font_manager import json_load


from report.API.html_generator_api import HtmlGeneratorApi


class SendReportData:
    """sends data to htmlGeneratorApi"""

    def __init__(self):
        self._filenames = {}

    @property
    def filenames(self):
        """prints out current plot filnames in list"""
        if len(self._filenames) < 1:
            print("No filenames to be sent")
        else:
            print("the following filnames is in list")
            for filename in self._filenames:
                print(filename)

    @filenames.setter
    def filenames(self, filenames):
        if not isinstance(filenames, dict):
            raise TypeError(
                "Send report: The filnames to be sent needs to be in a dict"
            )
        if len(filenames) < 1:
            raise ValueError("Send report: No filenames for images in filename list")
        self._filenames.update(filenames)

    def _img(self, filenames):
        return {
            "image_header": "Model Training",
            "image_location": f"training/{filenames['training']}",
            "about_image": " loss in the model ...",
        }

    def send(self, report_location="", report_filename=""):
        """send to api"""
        images = []
        path = "data/send.json"
        if "training" in self._filenames.keys():
            images.append(self._img(self._filenames))
            with open(path, "w", encoding="UTF-8") as file:
                json.dump(self._filenames, file)

        if "predictions" in self._filenames.keys():
            if os.path.isfile(path):
                with open(path, "r", encoding="UTF-8") as file:
                    images.append(self._img(json.load(file)))
            images.append(
                {
                    "image_header": "Model Prediction ",
                    "image_location": f"evaluation/{self._filenames['predictions']}",
                    "about_image": "lorem",
                }
            )
        if "confusion_matrix" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Confusion Matrix ",
                    "image_location": f"evaluation/{self._filenames['confusion_matrix']}",
                    "about_image": "lorem",
                }
            )
        if "classification_report" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Classification",
                    "image_location": f"evaluation/{self._filenames['classification_report']}",
                    "about_image": "lorem",
                }
            )

        if len(images) < 1:
            raise ValueError(
                "sendreport: unknown filnames skipping sending (not making any report)"
            )

        HtmlGeneratorApi(
            report_filename=report_filename,
            report_location=report_location,
            images=images,
        )
