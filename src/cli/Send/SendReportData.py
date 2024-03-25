import logging


from report.API.HtmlGeneratorApi import HtmlGeneratorApi


class SendReportData:
    """sends data to htmlGeneratorApi"""

    def __init__(self):
        self._filenames = []

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
    def filenames(self, filename_list):
        if not isinstance(filename_list, list):
            raise TypeError(
                "Send report: The filnames to be sent needs to be in a list"
            )
        if len(filename_list) < 1:
            raise ValueError("Send report: No filenames for images in filename list")
        self._filenames.extend(filename_list)

    def send(self, report_location="", report_filename=""):
        """send to api"""
        images = []
        i = 0
        for image in self._filenames:
            if "training" in image:
                images.append(
                    {
                        "image_header": "Model Training",
                        "image_location": f"training/{image['training']}",
                        "about_image": " loss in the model ...",
                    }
                )
            elif "predictions" in image:
                images.append(
                    {
                        "image_header": "Model Prediction ",
                        "image_location": f"evaluation/{image['predictions']}",
                        "about_image": "lorem",
                    }
                )
            elif "confusion_matrix" in image:
                images.append(
                    {
                        "image_header": "Confusion Matrix ",
                        "image_location": f"evaluation/{image['confusion_matrix']}",
                        "about_image": "lorem",
                    }
                )
            elif "classification_report" in image:
                images.append(
                    {
                        "image_header": "Classification",
                        "image_location": f"evaluation/{image['classification_report']}",
                        "about_image": "lorem",
                    }
                )
            else:
                if i == len(self._filenames) and len(images) < 1:
                    logging.warning(
                        "sendreport: unknown filnames skipping sending (not making any report)"
                    )
                    return
                else:
                    logging.warning(
                        "sendreport: Some of the filnames supplied in filnames list where unknown"
                    )
                i += 1
        HtmlGeneratorApi(
            report_filename=report_filename,
            report_location=report_location,
            images=images,
        )
