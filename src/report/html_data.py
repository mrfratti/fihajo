

class HtmlData:
    """Defines data about the html"""

    def __init__(self):
        self._html_store_location = ""
        self._filename = ""
        self._header_text = "Model Report"
        self._html_head = None
        self._menu = None
        self._main = None

    
    @property
    def head(self)->str:
        return self._html_head
    @head.setter
    def head(self,tag)->None:
        if isinstance(tag,str):
            self._html_head = tag
    @property
    def menu(self)->list:
        return self._menu
    @menu.setter
    def menu(self, menu):
        if isinstance(menu,list):
            self._menu=menu
    @property
    def main(self)->str:
        return self._main
    @main.setter
    def main(self, tag)->str:
        if isinstance(tag,str):
            self.main=tag
        
    @property
    def header_text(self) -> str:
        """header text for html report"""
        return self._header_text

    @header_text.setter
    def header_text(self, text):
        if not isinstance(text, str):
            raise TypeError("Header text for html document needs to be a string")
        if len(text) < 1:
            raise ValueError("Header text for html is missing")
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
            self.html_store_location = "src/report/reports/"
        else:
            self._html_store_location = location

    def __str__(self):
        return f"location set to {self._html_store_location} with filename {self._filename}"
