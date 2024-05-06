from src.report.html_data import HtmlData
from src.report.interactive.html_generator import InteractiveHtmlGenerator
from src.report.image_data import ImageData


class InteractiveHtmlGeneratorApi:
    """API for HtmlGenerator"""

    def __init__(self, report_location, report_filename, images):
        self._report = HtmlData()
        self._report.html_store_location = report_location
        self._report.filename = report_filename
        self._generator = InteractiveHtmlGenerator()
        self._generator.html_report = self._report
        for image in images:
            image_data = ImageData()
            image_data.header_image = image["image_header"]
            image_data.image_location = image["image_location"]
            image_data.about_image = image["about_image"]
            self._generator.image_data = image_data
        self._generator.write_html()




# class InteractiveHtmlGeneratorApi:

#     # def __init__(self, report_location, report_filename, images):
#     def __init__(self):
#         self._report = HtmlData()
#         self._generator = HtmlGenerator()
#         self._generator.html_report = self._report
#         self._report.head = "<div>SUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUp</div>"
#         self._report.main = "<div>Zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz</div>"
#         self._report.html_store_location="./"
#         self._report.filename = "test_index"

#         self._generator.write_html()

# if __name__ == "__main__":
#     InteractiveHtmlGeneratorApi()