from src.report.html_data import HtmlData
from src.report.html_generator import HtmlGenerator


class InteractiveHtmlGeneratorApi:

    # def __init__(self, report_location, report_filename, images):
    def __init__(self):
        self._report = HtmlData()
        self._report.head = "<div>"
        self._report.main = "<div>"
        self._report.filename = "test_index.html"
        self._generator = HtmlGenerator()
        self._generator.html_report = self._report
        self._generator.write_html()

if __name__ == "__main__":
    InteractiveHtmlGeneratorApi()