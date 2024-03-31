class HtmlData:
    """Defines data about the html"""

    def __init__(self):
        self._html_store_location = ""
        self._filename = ""
        self._header_text = "Model Report"

    @property
    def header_text(self) -> str:
        """header text for html report"""
        return self._header_text

    @header_text.setter
    def header_text(self, text):
        if len(text) < 1 or not isinstance(text, str):
            raise ValueError("Header text for html document needs to be a string")
        self._header_text = text

    @property
    def filename(self):
        """Returns current set filename for html report"""
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not isinstance(filename, str):
            raise TypeError("filename for html report needs to be a string")
        if len(filename) < 1:
            self._filename = "index.html"
        else:
            self._filename = f"{filename}.html"

    @property
    def html_store_location(self):
        """Returns current set store location for html report"""
        return self._html_store_location

    @html_store_location.setter
    def html_store_location(self, location):
        if not isinstance(location, str):
            raise TypeError("Store location needs string as input")
        if len(location) < 1:
            self.html_store_location = "./report/reports/"
        else:
            self._html_store_location = location

    def __str__(self):
        return f"location set to {self._html_store_location} with filename {self._filename}"
