from src.report.html_data import HtmlData
from src.report.html_generator import HtmlGenerator
from src.report.image_data import ImageData


class HtmlGeneratorApi:
    """API for HtmlGenerator"""

    def __init__(self, report_location, report_filename, images):
        self._report = HtmlData()
        self._report.html_store_location = report_location
        self._report.filename = report_filename
        self._generator = HtmlGenerator()
        self._generator.html_report = self._report
        for image in images:
            image_data = ImageData()
            image_data.header_image = image["image_header"]
            image_data.image_location = image["image_location"]
            image_data.about_image = image["about_image"]
            self._generator.image_data = image_data
        self._generator.write_html()
