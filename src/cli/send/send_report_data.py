from genericpath import exists
import json
import os

from report.API.html_generator_api import HtmlGeneratorApi


class SendReportData:
    """sends data to htmlGeneratorApi"""

    def __init__(self):
        self._filenames = {}
        self._path_json = "data/send.json"

    def delete_json(self):
        if os.path.exists(self._path_json):
            os.remove(self._path_json)

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
        existing = self._load_json()
        if existing is not None and len(existing) > 0:
            filenames.update(existing)
        with open(self._path_json, "w", encoding="UTF-8") as file:
            json.dump(filenames, file)

    def _load_json(self) -> dict:
        if os.path.isfile(self._path_json) and os.stat(self._path_json).st_size != 0:
            with open(self._path_json, "r", encoding="UTF-8") as file:
                return json.load(file)

    def _img(self, filenames):
        return {
            "image_header": "Model Training",
            "image_location": filenames["training"],
            "about_image": f" filename:{filenames['training']}",
        }

    def send(self, report_location="", report_filename=""):
        """send to api"""
        images = []
        self._filenames = self._load_json()
        if self._filenames is None or len(self._filenames) < 1:
            raise ValueError(
                "No plots to generate report from, run train, analyze or evaluate first."
            )
        if "training" in self._filenames.keys():
            images.append(self._img(self._filenames))

        if "adversarialTraining" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Adverserial Training",
                    "image_location": self._filenames["adversarialTraining"],
                    "about_image": "lorem",
                }
            )

        if "predictions" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Model Prediction ",
                    "image_location": self._filenames["predictions"],
                    "about_image": "lorem",
                }
            )
        if "confusion_matrix" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Confusion Matrix ",
                    "image_location": self._filenames["confusion_matrix"],
                    "about_image": "lorem",
                }
            )
        if "classification_report" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Classification",
                    "image_location": self._filenames["classification_report"],
                    "about_image": "lorem",
                }
            )
        if "pcs_meansoftmax" in self._filenames.keys():
            images.append(
                {
                    "image_header": "PCS Mean-soft-max",
                    "image_location": self._filenames["pcs_meansoftmax"],
                    "about_image": "lorem",
                }
            )
        if "distrubution_meansoftmax" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Distribution of Mean-soft-max",
                    "image_location": self._filenames["distrubution_meansoftmax"],
                    "about_image": "lorem",
                }
            )
        if "pcs_inverse" in self._filenames.keys():
            images.append(
                {
                    "image_header": "PCS Inverse",
                    "image_location": self._filenames["pcs_inverse"],
                    "about_image": "lorem",
                }
            )
        if "entropy_distrubution" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Entropy distribution",
                    "image_location": self._filenames["entropy_distrubution"],
                    "about_image": "lorem",
                }
            )
        if "higly_uncertain_inputs" in self._filenames.keys():
            images.append(
                {
                    "image_header": "Highly uncertain inputs",
                    "image_location": self._filenames["higly_uncertain_inputs"],
                    "about_image": "lorem",
                }
            )
        if "prediction_vs_entrophy" in self._filenames.keys():
            images.append(
                {
                    "image_header": "prediction vs entropy",
                    "image_location": self._filenames["prediction_vs_entrophy"],
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
        self.delete_json()
