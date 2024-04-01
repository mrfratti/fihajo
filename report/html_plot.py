from matplotlib import pyplot as plt
from mpld3 import fig_to_html, plugins
import numpy as np


class HtmlPlot:
    def __init__(self) -> None:
        self._fig, self._ax = plt.subplots(subplot_kw=dict(facecolor="#EEEEEE"))
        self._fig.set_size_inches(7, 7)
        self._header = "Interactive plot"

    @property
    def plot(self) -> str:
        """Returns html code of an plot"""
        plot = ""
        for line in fig_to_html(self._fig).splitlines():
            if -1 == line.find("//") or -1 != line.find("https://"):
                plot += line.replace("<style>", "").replace("</style>", "")
        return plot

    @plot.setter
    def plot(self, sizeN=100):
        scatter = self._ax.scatter(
            np.random.normal(size=sizeN),
            np.random.normal(size=sizeN),
            c=np.random.random(size=sizeN),
            s=1000 * np.random.random(size=sizeN),
            alpha=0.3,
        )
        self._ax.grid(color="white", linestyle="solid")
        self._ax.set_title("Scatter Plot Example", size=20)
        labels = ["point {0}".format(i + 1) for i in range(sizeN)]
        tooltip = plugins.PointLabelTooltip(scatter, labels=labels)
        plugins.connect(self._fig, tooltip)

    @property
    def header(self):
        return self._header
