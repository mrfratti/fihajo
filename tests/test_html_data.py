import unittest

from src.report.html_data import HtmlData


class TestHtmlData(unittest.TestCase):
    def setUp(self) -> None:
        self.html_data = HtmlData()

    def test_header_should_be_string(self):
        with self.assertRaises(TypeError):
            self.html_data.header_text = 3

    def test_header_length(self):
        with self.assertRaises(ValueError):
            self.html_data.header_text = ""

    def test_default_store_location(self):
        default = "src/report/reports/"
        self.html_data.html_store_location = ""
        self.assertEqual(default, self.html_data.html_store_location)
