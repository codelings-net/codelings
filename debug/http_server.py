#!/usr/bin/python3

import http.server, socketserver

PORT = 4096

try:
	Handler = http.server.SimpleHTTPRequestHandler
	Handler.extensions_map[".wasm"] = "application/wasm"
	httpd = socketserver.TCPServer(("", PORT), Handler)
except:
	raise

print("serving at port", PORT)
httpd.serve_forever()
