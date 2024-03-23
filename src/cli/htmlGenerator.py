import datetime
from yattag import Doc


doc, tag, text = Doc().tagtext()


class htmlReport:

    # Contstructor
    def __init__(self, inputApi=[]):
        self._inputApi = inputApi

    def generate(self):
        with tag("html"):
            with tag("head"):
                doc.stag("link", rel="stylesheet", href="../report/style.css")
            with tag("body"):
                with tag("main"):
                    if len(self._inputApi) < 1:
                        with tag("h1"):
                            text("no data")
                    else:
                        for data in self._inputApi:
                            with tag("div", klass="section"):
                                with tag("h1"):
                                    text(data["header"])
                                doc.stag("img", src=data["image"], klass="photo")
                                with tag("p"):
                                    text(data["text"])
        return doc.getvalue()

    @property
    def setData(self, data):
        self._data = data

    def writeHtml(self, filename="report"):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{filename}_{timestamp}.html"
            file = open(f"./report/{filename}", "w", encoding="UTF()")
            file.write(self.generate())
            file.close()
        except Exception as e:
            print(f"En feil oppstod {e}")
