import unittest

from tomlkit import value
from report.image_data import ImageData


class TestImageData(unittest.TestCase):
    def setUp(self):
        self.image = ImageData()

    def test_should_raise_type_error_for_non_string(self):
        with self.assertRaises(TypeError):
            self.image.image_location = 3
            self.image.about_image = 2
            self.image.header_image = 2

    def test_should_raise_value_error_when_empty_string(self):
        with self.assertRaises(ValueError):
            self.image.image_location = ""


if __name__ == "__main__":
    unittest.main()
