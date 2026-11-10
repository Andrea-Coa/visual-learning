"""Flask wrapper for the static site in app/ (PLAN 8.5).

The site needs no server code: every data file is precomputed JSON in app/data/. This app only serves the folder,
so the Canvas ZIP contains the Flask application the assignment asks for.

Usage:
    pip install flask
    python app.py                 # then open http://127.0.0.1:5000
"""
from pathlib import Path

from flask import Flask

SITE = Path(__file__).resolve().parent / "app"

app = Flask(__name__, static_folder=str(SITE), static_url_path="")


@app.get("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
