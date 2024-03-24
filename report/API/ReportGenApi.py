from report.HtmlData import HtmlData
from report.HtmlGenerator import HtmlGenerator
from report.ImageData import ImageData


class ReportGenApi:
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
        self._generator.writeHtml()
