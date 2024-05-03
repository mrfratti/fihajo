import unittest

from src.report.image_data import ImageData


class TestImageData(unittest.TestCase):
    def setUp(self):
        self.image = ImageData()

    def test_image_location_should_be_string(self):
        with self.assertRaises(TypeError):
            self.image.image_location = 1

    def test_about_image_should_be_string(self):
        with self.assertRaises(TypeError):
            self.image.about_image = 2

    def test_image_header_should_be_string(self):
        with self.assertRaises(TypeError):
            self.image.header_image = 3

    def test_image_location_should_not_empty(self):
        with self.assertRaises(ValueError):
            self.image.image_location = ""


if __name__ == "__main__":
    unittest.main()
