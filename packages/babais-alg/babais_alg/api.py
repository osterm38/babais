from .app import create
from .settings import settings

def start(
    port: int = settings.port,
):
    print('starting...(not implemented)')
    app = create()

def stop():
    print('stopping...(not implemented)')