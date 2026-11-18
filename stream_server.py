"""Serve annotated frames as an MJPEG stream so a browser on another computer can watch."""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><title>Crack vision</title>
<body style="margin:0;background:#111"><img src="/stream" style="width:100%;max-width:960px;display:block;margin:auto">"""


class MjpegServer:
    def __init__(self, port, quality=70):
        self.quality = quality
        self.jpeg, self.version = None, 0
        self.condition = threading.Condition()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    seen = 0
                    try:
                        while True:
                            with owner.condition:
                                owner.condition.wait_for(lambda: owner.version != seen, timeout=5)
                                jpeg, seen = owner.jpeg, owner.version
                            if jpeg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                            self.wfile.write(jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass
                elif self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update(self, frame):
        import cv2
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.condition:
                self.jpeg, self.version = encoded.tobytes(), self.version + 1
                self.condition.notify_all()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
