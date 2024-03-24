class ImageData:
    def __init__(self):
        self._image_location = ""
        self._about_image = ""
        self._header_image = ""

    @property
    def image_location(self):
        return self._image_location

    @property
    def about_image(self):
        return self._about_image

    @property
    def header_image(self):
        return self._header_image

    @image_location.setter  # noqa: F821
    def image_location(self, location):
        if not isinstance(location, str):
            raise TypeError("Image location needs to be a string")
        if len(location) < 1:
            raise ValueError("Missing input for image location")
        self._report_store_location = location

    @about_image.setter  # noqa: F821
    def about_image(self, text):
        if not isinstance(text, str):
            raise TypeError("info about the image needs to be a string")
        self._report_store_location = text

    @header_image.setter  # noqa: F821
    def header_image(self, text):
        if not isinstance(text, str):
            raise TypeError("header for image needs to be a string")
        self._report_store_location = text
