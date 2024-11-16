from .app import create
from .settings import settings

def start(
    port: int = settings.port,
):
    print('starting...(not implemented)')
    app = create()
    try:
        app.launch(server_port=port)
    except Exception as e:
        pass

def stop():
    print('stopping...(not implemented)')