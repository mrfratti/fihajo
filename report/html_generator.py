import logging
import os
from yattag import Doc
from report.html_data import HtmlData
from report.html_plot import HtmlPlot
from report.image_data import ImageData
from src.cli.string_styling import StringStyling


doc, tag, text = Doc().tagtext()


class HtmlGenerator:
    """Generates html report"""

    def __init__(self) -> None:
        self._image_data_list = []
        self._html_report = HtmlData()
        self._html_plot = HtmlPlot()
        self._html_plot.plot = 100

    @property
    def image_data(self) -> int:
        """Returns a count of images"""
        return len(self._image_data_list)

    @image_data.setter
    def image_data(self, data):
        if data is None or not isinstance(data, ImageData):
            raise ValueError("Wrong data type when adding image_data")
        self._image_data_list.append(data)

    @property
    def html_report(self) -> None:
        """Prints out info about html"""
        print(StringStyling.box_style(self._html_report))

    @html_report.setter
    def html_report(self, report):
        if report is None or not isinstance(report, HtmlData):
            raise ValueError("HtmlReport needs htmldata type, wrong datatype set")
        self._html_report = report

    def _generate(self) -> str:
        doc.asis("<!DOCTYPE html>")
        with tag("html"):
            with tag("head"):
                with tag("meta", charset="UTF-8"):
                    pass
                doc.stag("link", rel="stylesheet", href="dist/style.css")
                with tag("meta", http_equiv="Content-Security-Policy"):
                    doc.attr(
                        content="default-src 'self'; font-src 'self' http://localhost:8090;"
                    )
                with tag(
                    "meta",
                    name="viewport",
                    content="width=device-width, initial-scale=1.0",
                ):
                    pass

            with tag("body"):
                with tag("header"):
                    with tag("h1"):
                        text(self._html_report.header_text)
                with tag("main"):
                    self._main()

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
            self._plot()

    def _img(self):
        for data in self._image_data_list:
            with tag("div", klass="section"):
                with tag("h2"):
                    text(data.header_image)
                doc.stag("img", src=data.image_location, klass="photo")
                with tag("p"):
                    text(data.about_image)
                with tag("a", href=data.image_location):
                    text("Image link")

    def _plot(self):
        with tag("div", klass="section"):
            with tag("h2"):
                text(self._html_plot.header)
            doc.asis(self._html_plot.plot)

    def write_html(self) -> None:
        """Writes the html file when the html is generated"""
        try:
            if len(self._image_data_list) < 1:
                logging.warning(StringStyling.box_style("No images where supplied"))
            if not os.path.exists(self._html_report.html_store_location):
                os.mkdir(self._html_report.html_store_location)
                logging.info(
                    "Htmlgenerator: Dirctory %s did not exist making directory",
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
            print(
                StringStyling.box_style("cannot open html file or filepath not found")
            )
        except PermissionError:
            print(StringStyling.box_style("Missing permission to write html file"))
