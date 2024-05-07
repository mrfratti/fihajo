class InteractiveImageData:
    """defines data for an image"""

    def __init__(self):
        self._image_location = ""
        self._about_image = ""
        self._header_image = ""

    @property
    def image_location(self):
        """returns the current set image location"""
        return self._image_location

    @property
    def about_image(self):
        """returns about description for an image"""
        return self._about_image

    @property
    def header_image(self):
        """returns the header for an image"""
        return self._header_image

    @image_location.setter
    def image_location(self, location):
        if not isinstance(location, str):
            raise TypeError("Image location needs to be a string")
        if len(location) < 1:
            raise ValueError("Missing input for image location")
        # rm = "/src/report/reports"
        rm = ""
        if location.find(rm):
            pos = rm.index(rm) + len(rm)
            location = location[pos:]
        self._image_location = location

    @about_image.setter
    def about_image(self, text):
        if not isinstance(text, str):
            raise TypeError("info about the image needs to be a string")
        self._about_image = text

    @header_image.setter
    def header_image(self, text):
        if not isinstance(text, str):
            raise TypeError("header for image needs to be a string")
        self._header_image = text
