# webserver for patent search

NUM_RESULTS = 10

import http.server
import socketserver
import urllib
import json

from model2 import findSimilarPatents

PORT = 8888

class Server(socketserver.TCPServer):
    allow_reuse_address = True

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
  def do_GET(self):
    self.path = '/static'+self.path
    return http.server.SimpleHTTPRequestHandler.do_GET(self)
  def do_POST(self):
    if self.path == '/api/search':
      self.send_response(200)
      self.send_header("Content-type", "application/json")
      self.end_headers()

      # print(dir(self))
      self.query_string = self.rfile.read(int(self.headers['Content-Length'])).decode()
      self.args = dict(urllib.parse.parse_qsl(self.query_string))

      query = self.args['q']

      print(query)

      results = findSimilarPatents(query, NUM_RESULTS)
      self.wfile.write(json.dumps(results).encode())

      # f = open('static/demoResults.json', 'rb')
      # self.wfile.write(f.read())
      # f.close()

    else:
      self.send_response(404)
      self.end_headers()
      self.wfile.write(b'no')

Handler = MyHttpRequestHandler

with Server(("", PORT), Handler) as httpd:
  print("serving at port", PORT)
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    httpd.shutdown()
    httpd.server_close()
