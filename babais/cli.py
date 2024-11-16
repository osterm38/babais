from typer import Typer
from babais_web.cli import app as web
from babais_alg.cli import app as alg


app = Typer()

app.add_typer(web, name='web')
app.add_typer(alg, name='alg')

if __name__ == "__main__":
    """can run this with `python -m babais.cli --help"""
    app()