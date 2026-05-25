"""BassLift one-click launcher — start server + open browser."""
import threading
import time
import webbrowser
import sys

import uvicorn

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}/"


def open_browser_when_ready():
    # Daj uvicornowi chwilę na bind portu, potem otwórz przeglądarkę
    time.sleep(1.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    print(f"BassLift uruchamia się pod {URL}")
    print("Aby zatrzymać: Ctrl+C lub zamknij to okno.\n")
    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    try:
        uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="info")
    except KeyboardInterrupt:
        sys.exit(0)
