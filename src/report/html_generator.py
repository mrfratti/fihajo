import logging
import os
from yattag import Doc
from src.report.html_data import HtmlData
from src.report.image_data import ImageData
from src.cli.string_styling import StringStyling

doc, tag, text = Doc().tagtext()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message).80s", level=logging.INFO)


class HtmlGenerator:
    """Generates html report"""

    def __init__(self) -> None:
        self._image_data_list = []
        self._html_report = HtmlData()
        self._order = {
            "training": 1,
            "predictions": 2,
            "confusion_matrix": 3,
            "classification_report": 4,
            "adversarial_examples": 5,
            "accuracy_comparison": 6,
            "pcs_meansoftmax": 7,
            "distribution_meansoftmax": 8,
            "pcs_inverse": 9,
            "entropy_distribution": 10,
            "highly_uncertain_inputs": 11,
            "prediction_vs_entropy": 12
        }

    @property
    def image_data(self) -> int:
        return len(self._image_data_list)

    @image_data.setter
    def image_data(self, data):
        if data is None or not isinstance(data, ImageData):
            raise ValueError("Wrong data type when adding image_data")
        self._image_data_list.append(data)

    def _generate(self) -> str:
        doc, tag, text = Doc().tagtext()
        doc.asis("<!DOCTYPE html>")
        with tag("html"):
            with tag("head"):
                doc.stag("meta", charset="UTF-8")
                doc.stag("link", rel="stylesheet", href="dist/style.css")
                doc.stag("meta", name="viewport", content="width=device-width, initial-scale=1.0")
            with tag("body"):
                with tag("header"):
                    with tag("h1"):
                        text(self._html_report.header_text)
                with tag("main"):
                    self._main()
                with tag("footer"):
                    text("Copyright ©")
        return doc.getvalue()

    def _main(self):
        self._image_data_list.sort(key=lambda x: self._order.get(x.header_image.lower(), 99))
        if not self._image_data_list:
            with tag("div", klass="error"):
                with tag("h2"):
                    text("Oops!")
                with tag("p"):
                    text("No data to show")
        else:
            self._img()

    def _img(self):
        for i, data in enumerate(self._image_data_list):
            if i % 2 == 0:
                self._img_section(data, "right")
            else:
                self._img_section(data, "left")

    def _img_section(self, data, section):
        with tag("div", klass=section):
            with tag("h2"):
                text(data.header_image)
            doc.stag("img", src=data.image_location)
            with tag("div", klass="info"):
                with tag("p"):
                    text(data.about_image)
                with tag("a", href=data.image_location, target="_blank"):
                    with tag("button"):
                        text("Open Image File")

    def write_html(self) -> None:
        try:
            if not self._image_data_list:
                logging.warning("No images were supplied")
            os.makedirs(self._html_report.html_store_location, exist_ok=True)
            filepath = os.path.join(self._html_report.html_store_location, self._html_report.filename)
            with open(filepath, "w", encoding="UTF-8") as file:
                file.write(self._generate())
                logging.info("Report written to %s", filepath)
        except Exception as e:
            logging.error("An error occurred while writing the HTML file: %s", e)
