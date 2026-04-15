"""Example Python plugin for EOV.

This script is spawned as a subprocess by the EOV host when the plugin's
toolbar button is clicked. It uses slint-python to load the .slint UI file,
wire callbacks, and run the event loop with its own window.

The host invokes this script using the .venv/bin/python3 interpreter inside
the plugin directory, so all dependencies (e.g. slint) are available without
modifying the system Python installation.
"""

import os
import slint

script_dir = os.path.dirname(os.path.abspath(__file__))
components = slint.load_file(os.path.join(script_dir, "ui", "my_panel.slint"))

panel = components.MyPanel()


def on_greet_clicked():
    print("Button clicked in Python plugin!")


def on_clear_clicked():
    print("[python_plugin] Clear clicked")


panel.greet_clicked = on_greet_clicked
panel.clear_clicked = on_clear_clicked

panel.run()
