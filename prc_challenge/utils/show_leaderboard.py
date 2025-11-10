import json
import ipywidgets as widgets
from IPython.display import display

from prc_challenge.utils import collect_experiments


def show_leaderboard():

    # Collect and sort experiments by RMSE ascending (best to worst)
    experiments = collect_experiments()
    experiments.sort(key=lambda x: x['rmse'])

    # Create Accordion widget for the leaderboard
    accordion = widgets.Accordion()

    # List to hold content widgets
    children = []

    for exp in experiments:
        # Title: short_name - RMSE: value
        title = f"{exp['short_name']} - RMSE: {exp['rmse']:.4f}"
        
        # Content: Seed and formatted config
        content_html = (
            f"<pre>"
            f"Timestamp: {exp['timestamp']}\n"
            f"Seed: {exp['seed']}\n"
            f"Config:\n"
            f"{json.dumps(exp['config'], indent=2)}"
            f"</pre>"
        )
        content = widgets.HTML(value=content_html)
        
        children.append(content)

    # Set children and titles
    accordion.children = children
    for i, exp in enumerate(experiments):
        accordion.set_title(i, f"{exp['short_name']} - RMSE: {exp['rmse']:.4f}")

    # Display the widget
    display(accordion)
