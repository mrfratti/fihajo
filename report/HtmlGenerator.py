import datetime
from email import message
import logging
from yattag import Doc
from report.HtmlData import HtmlData
from report.ImageData import ImageData
from src.cli.stringformat import String_Format as format


doc, tag, text = Doc().tagtext()


class HtmlGenerator:
    def __init__(self):
        self._image_data_list = []
        self._html_report = HtmlData()

    @property
    def image_data(self):
        """Returns a count of images"""
        return len(self._image_data_list)

    @image_data.setter
    def image_data(self, data):
        if data is None or not isinstance(data, ImageData):
            raise ValueError("Wrong data type when adding image_data")
        self._image_data_list.append(data)

    @property
    def html_report(self):
        print(format.message(self._html_report))

    @html_report.setter
    def html_report(self, report):
        if report is None or not isinstance(report, HtmlData):
            raise ValueError("HtmlReport needs htmldata type, wrong datatype set")
        self._html_report = report

    def _generate(self):
        with tag("html"):
            with tag("head"):
                doc.stag("link", rel="stylesheet", href="../style.css")
            with tag("body"):
                with tag("main"):
                    if len(self._image_data_list) < 1:
                        with tag("div", klass="error"):
                            with tag("h1"):
                                text("Oops!")
                            with tag("p"):
                                text("No data to show")
                    else:
                        for data in self._image_data_list:
                            with tag("div", klass="section"):
                                with tag("h1"):
                                    text(data.header_image)
                                doc.stag("img", src=data.image_location, klass="photo")
                                with tag("p"):
                                    text(data.about_image)
        return doc.getvalue()

    def writeHtml(self):
        try:
            if len(self._image_data_list) < 1:
                logging.warning(format.message("No images where supplied"))
            file = open(
                f"{self._html_report.html_store_location}{self._html_report.filename}",
                "w",
                encoding="UTF()",
            )
            logging.info(format.message("Writing report"))
            file.write(self._generate())
            file.close()
        except ValueError as e:
            logging.warning(e)
            pass
        except TypeError as e:
            logging.warning(e)
            pass
        except Exception as e:
            logging.critical(format.message(f"An error occoured: {e}"))
