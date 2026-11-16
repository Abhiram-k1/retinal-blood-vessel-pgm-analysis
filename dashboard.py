"""
Interactive Web Dashboard Launcher for Retinal Vascular Network Analysis.
Launches a local HTTP server serving the visual dashboard and opens it in your default web browser.
"""

import http.server
import os
import socketserver
import sys
import threading
import time
import webbrowser

PORT = 8080
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Suppress standard noisy logging
        pass


def find_free_port(start_port=8080, max_attempts=20):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    if not os.path.exists(DIRECTORY):
        print(f"Error: Directory '{DIRECTORY}' does not exist.")
        sys.exit(1)

    port = find_free_port(PORT)
    url = f"http://localhost:{port}/index.html"

    print("=" * 70)
    print("  RETINAL VASCULAR NETWORK ANALYSIS - INTERACTIVE RESULTS DASHBOARD")
    print("=" * 70)
    print(f"  Serving directory : {DIRECTORY}")
    print(f"  Dashboard URL     : {url}")
    print("=" * 70)
    print("  Opening browser...")

    server = socketserver.TCPServer(("127.0.0.1", port), Handler)

    # Open browser in a separate thread after server starts listening
    def open_browser():
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"  Dashboard is live! Press Ctrl+C in terminal to stop server.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down dashboard server.")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
