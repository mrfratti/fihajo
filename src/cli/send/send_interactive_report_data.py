import json
import os

from src.report.API.html_generator_api import HtmlGeneratorApi
from src.report.image_data import ImageData

class SendInteractiveReportData:
    """sends data to htmlGeneratorApi"""

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Put inn data path in list too from visualtitation
    # orgin report is used in each funtion(main.py): training, evaluate ..... 

    def __init__(self):
        self._filenames = {}
        self.adversarial_evaluated = False
        self._path_json = "data/send_interactive.json"
    
    def delete_json(self):
        if os.path.exists(self._path_json):
            os.remove(self._path_json)

    @property
    def filenames(self):
        """prints out current plot filnames in list"""
        if len(self._filenames) < 1:
            print("No filenames to be sent")
        else:
            print("the following filenames is in list")
            for filename in self._filenames:
                print(filename)

    @property
    def adversarial_evaluated(self):
        return self._adversarial_evaluated

    @adversarial_evaluated.setter
    def adversarial_evaluated(self, value: bool):
        self._adversarial_evaluated = value

    @filenames.setter
    def filenames(self, filenames):
        if not isinstance(filenames, dict):
            raise TypeError("Send report: The filenames to be sent needs to be in a dict")
        if len(filenames) < 1:
            raise ValueError("Send report: No filenames for images in filename list")
        existing = self._load_json()
        if existing is not None and len(existing) > 0:
            filenames.update(existing)
        print("Writing filename to JSON:", filenames)
        with open(self._path_json, "w", encoding="UTF-8") as file:
            json.dump(filenames, file)

    def _load_json(self) -> dict:
        if os.path.isfile(self._path_json) and os.stat(self._path_json).st_size != 0:
            with open(self._path_json, "r", encoding="UTF-8") as file:
                data = json.load(file)
                print("Loaded filenames from JSON:", data)
                return data
                #return json.load(file)
        return {}

    def _img(self, key, filenames):
        """Preparing image data for HTML report"""
        descriptions = {
            "training": (
                "Model Training",
                "This plot displays the training accuracy and loss over each epoch, providing insights into the "
                "model's learning progress and convergence behavior.",
            ),
            "adversarialTraining": (
                "Adversarial Training",
                "Insights into model performance over each epoch providing when exposed to adversarially modified "
                "inputs."
            ),
            "predictions": (
                "Model Predictions",
                "Displays predictions from the model alongside actual labels, highlighting how well the model "
                "performs on unseen data."
            ),
            "confusion_matrix": (
                "Confusion Matrix",
                "This confusion matrix provides a detailed breakdown of the model's predictions across different "
                "classes, helping identify classes that are frequently confused."
            ),
            "classification_report": (
                "Classification Report",
                "Presents a comprehensive report of precision, recall, and F1-scores for each class. Precision "
                "measures the accuracy of positive predictions for each class. It's defined as the number of true "
                "positive predictions divided by the total number og positive predictions (true positives plus false "
                "positives), e.g., for digit 0, a precision of 0.98 means that 98% of the model's prediction for "
                "digit 0 were correct. Recall (also known as sensitivity) measures the ability if the model to find "
                "all the relevant cases within a dataset. It's defined as the number of true positive predictions "
                "divided by the total number of actual positives (true positives plus false negatives). E.g., "
                "for digit 0, a recall of 0.99 means the model correctly identified 99% if all actual 0s in the "
                "dataset. F1-Score is a measure of a model's accuracy that combines precision and recall into a "
                "single metric by taking their harmonic mean. It's useful when we need to balance precision and "
                "recall. An F1-score of 0.99 for digit 0 indicates a very good balance between precision and recall "
                "for this class."
            ),
            "accuracy_comparison": (
                "Accuracy Comparison",
                "Bar graph comparing model accuracy on clean vs. adversarially altered data. "
                "Shows how adversarial attack performed on data that has not been adversarial trained, will impact"
                "models predictions"
            ),
            "adversarial_examples": (
                "Adversarial Examples",
                "Visualizes the outputs of adversarial examples and shows the model's predictions, highlighting the "
                "vulnerability and robustness of the model under attack."
            ),
            "pcs_meansoftmax": (
                "PCS and Mean Softmax",
                "Graphs of Predictive Confidence Score and Mean Softmax outputs, indicating confidence levels and "
                "class probabilities."
            ),
            "distrubution_meansoftmax": (
                "Distribution of Mean Softmax",
                "Distribution plot of softmax output values, providing insights into model certainty across classes."
            ),
            "pcs_inverse": (
                "PCS Inverse",
                "Plot of inverse Predictive Confidence Scores, useful for understanding areas of high model "
                "uncertainty."
            ),
            "entropy_distrubution": (
                "Entropy Distribution",
                "Visualization of entropy in model predictions, highlighting areas where the model is least certain."
            ),
            "higly_uncertain_inputs": (
                "Highly Uncertain Inputs",
                "Showcases inputs for which the model exhibited the highest uncertainty, useful for further analysis."
            ),
            "prediction_vs_entrophy": (
                "Prediction vs. Entropy",
                "Scatter plot analyzing the relationship between model predictions and their associated entropy."
            )
        }
        header, about = descriptions.get(key, (key.replace("_", " ").title(), f"Generated plot for "
                                                                              f"{key.replace('_', ' ')}"))
        if key in filenames:
            return {
                "image_header": header,
                "image_location": filenames[key],
                "about_image": about
            }
        else:
            print(f"Key {key} not found in filenames")

    def send(self, report_location="", report_filename=""):
        """Send to API"""
        images = []
        self._filenames = self._load_json()

        for i in range(len(self._filenames)-1, -1, -1):
            img_data = self._img(list(self._filenames.keys())[i], self._filenames)
            if img_data is not None:
                images.append(img_data)

        if len(images) < 1:
            raise ValueError("No images to generate report from, run train, evaluate and analyze to before generating "
                             "a report!")

        HtmlGeneratorApi(
            report_filename=report_filename,
            report_location=report_location,
            images=images,
        )
        self.delete_json()
