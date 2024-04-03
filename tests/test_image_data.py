import unittest
from report.image_data import ImageData


class TestImageData(unittest.TestCase):
    def setUp(self):
        self.image = ImageData()

    def test_should_return_string(self):
        with self.assertRaises(TypeError):
            self.image.image_location = 3


if __name__ == "__main__":
    unittest.main()
