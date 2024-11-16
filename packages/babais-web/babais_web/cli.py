from typer import Typer
from .api import start, stop

app = Typer()

app.command()(start)
app.command()(stop)

if __name__ == "__main__":
    """can run this with `python -m babais_web.cli --help"""
    app()