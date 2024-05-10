from src.report.html_data import HtmlData
from src.report.interactive.html_generator import InteractiveHtmlGenerator
from src.report.interactive.image_data import InteractiveImageData


class InteractiveHtmlGeneratorApi:

    def __init__(self, report_location, report_filename, images):
        self._report = HtmlData()
        self._generator = InteractiveHtmlGenerator()
        self._generator.html_report = self._report
        text2 = report_filename

        for image in images:
            image_data = InteractiveImageData()
            image_data.header_image = image["image_header"]
            image_data.image_location = image["image_location"]
            image_data.about_image = image["about_image"]
            self._generator.image_data = image_data

            
        self._report.html_store_location="./src/report/reports/"
        self._report.filename = "index_interactive"

        self._generator.write_html()
