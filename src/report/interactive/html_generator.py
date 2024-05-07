import logging
import os
from yattag import Doc
from src.report.html_data import HtmlData
from src.report.image_data import ImageData
from src.cli.string_styling import StringStyling


doc, tag, text = Doc().tagtext()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message).80s", level=logging.INFO)


class InteractiveHtmlGenerator:
    """Generates html report"""

    def __init__(self) -> None:
        self._image_data_list = []
        self._html_report = HtmlData()

    @property
    def image_data(self) -> int:
        """Returns a count of images"""
        return len(self._image_data_list)

    @image_data.setter
    def image_data(self, data):
        # if data is None or not isinstance(data, ImageData):
        #     raise ValueError("Wrong data type when adding image_data")
        self._image_data_list.append(data)

    @property
    def html_report(self) -> None:
        """Prints out info about html"""
        print(StringStyling.box_style(self._html_report))

    @html_report.setter
    def html_report(self, report):
        if report is None or not isinstance(report, HtmlData):
            raise ValueError("HTMLReport needs HTMLData type, wrong datatype set")
        self._html_report = report

    def _generate(self):
        doc.asis("<!DOCTYPE html>")
        with tag("html", lang="en"):
            with tag("head"):
                doc.stag("meta", charset="UTF-8")
                doc.stag("link", rel="stylesheet", href="src/report/reports/dist/style.css")
                doc.stag("meta", name="viewport", content="width=device-width, initial-scale=1.0")

            with tag("body"):
                with tag("header"):
                    with tag("h1"):
                        text(self._html_report.header_text)
                    with tag("nav"):
                        with tag("ul"):
                            for i, data in enumerate(self._image_data_list):
                                with tag("li"):
                                    with tag("a", href=f"#section{i}"):
                                        text(data.header_image)

                with tag("main"):
                    for i, data in enumerate(self._image_data_list):
                        with tag("section", id=f"section{i}"):
                            self._img_section(data, "right" if i % 2 == 0 else "left")

                with tag("footer"):
                    text("Copyright © Firat Celebi, Joakim Hole Polden, Harykaran Lambotharan")
        return doc.getvalue()

    def _main(self):
        if len(self._image_data_list) < 1:
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

        file_path = data.image_location
        file_split = os.path.splitext(file_path)
        file_end_lower = file_split[1].lower()

        with tag("div", klass=section):
            with tag("h2"):
                text(data.header_image)

            if file_end_lower == '.json':   
                with tag("iframe", src=data.image_location, style="width:100%; height:400px; border:none;"):
                    text("Your browser does not support iframes")

            elif file_end_lower == '.html':
                with tag("iframe", src=data.image_location, style="width:100%; height:400px; border:none;"):
                    text("Your browser does not support iframes")
            
            elif file_end_lower == '.png':
                with tag('img', src=file_path, style="width:auto; height:auto;"):
                    text('Your browser does not support images')

            with tag("div", klass="info"):
                with tag("div"):
                    with tag("p"):
                        text(data.about_image)
                with tag("a", href=data.image_location):
                    with tag("button"):
                        text("Open Image File")

    def write_html(self) -> None:
        """Writes the html file when the html is generated"""
        try:
            if len(self._image_data_list) < 1:
                logging.warning(StringStyling.box_style("No images where supplied"))
            if not os.path.exists(self._html_report.html_store_location):
                os.mkdir(self._html_report.html_store_location)
                logging.info(
                    "Htmlgenerator: Directory %s did not exist making directory",
                    self._html_report.html_store_location,
                )
            with open(
                f"{self._html_report.html_store_location}{self._html_report.filename}",
                "w",
                encoding="UTF-8",
            ) as file:
                logging.info(StringStyling.box_style("Writing report"))
                file.write(self._generate())
                file.close()
        except ValueError as e:
            logging.warning("htmlgenerator: %s", e)
            return
        except TypeError as e:
            logging.warning("htmlgenerator: %s", e)
            return
        except FileNotFoundError:
            print(StringStyling.box_style("Cannot open html file or filepath not found"))
        except PermissionError:
            print(StringStyling.box_style("Missing permission to write html file"))