"""
Helpers: save_figure
"""

from pathlib import Path
import plotly.graph_objects as go

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

'''
----SAVE_FIGURE----
INPUT: go.figure, name (str)
OUTPUT: figure saved to results/
'''
def save_figure(fig: go.Figure, name: str) -> None:
    # write fig to /results as interactive html and static png
    fig.write_html(RESULTS_DIR / f"{name}.html", include_plotlyjs='cdn')
    fig.write_image(RESULTS_DIR / f"{name}.png", width=1200, height=700, scale=2)