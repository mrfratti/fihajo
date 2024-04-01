from matplotlib import pyplot as plt
from mpld3 import fig_to_html, plugins
import numpy as np


class HtmlPlot:
    def __init__(self) -> None:
        self._fig, self._ax = plt.subplots(subplot_kw=dict(facecolor="#EEEEEE"))

    @property
    def plot(self) -> str:
        """Returns html code of an plot"""
        return str(fig_to_html(self._fig))

    @plot.setter
    def plot(self):
        N = 100
        scatter = self._ax.scatter(
            np.random.normal(size=N),
            np.random.normal(size=N),
            c=np.random.random(size=N),
            s=1000 * np.random.random(size=N),
            alpha=0.3,
        )
        self._ax.grid(color="white", linestyle="solid")
        self._ax.set_title("Scatter Plot Example", size=20)
        labels = ["point {0}".format(i + 1) for i in range(N)]
        tooltip = plugins.PointLabelTooltip(scatter, labels=labels)
        plugins.connect(self._fig, tooltip)
