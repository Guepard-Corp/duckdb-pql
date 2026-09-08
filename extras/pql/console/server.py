#!/usr/bin/env python3
"""Minimal browser console for this DuckDB build, with PQL.

    python3 extras/pql/console/server.py [db_path] [port]

Keeps ONE duckdb process alive for the whole session, which matters: PQL models
live in memory per database instance, so a TRAIN in one request must still be
visible to a PREDICT in the next. Binds to localhost only.
"""
import json, os, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(ROOT, "..", "..", "..", "build", "reldebug", "duckdb")
DB = sys.argv[1] if len(sys.argv) > 1 else ":memory:"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 4213
SENTINEL = "__pql_done__"

class Session:
    """Runs a whole submission in one duckdb invocation.

    PQL models live in memory for the life of a database instance, so a TRAIN
    and the PREDICT that uses it must travel together. Keeping a CLI alive
    across requests is possible but brittle (the CLI buffers all of stdin when
    it is not on a terminal), and one process per submission is both simpler and
    harder to wedge. Write the whole script in the box.
    """
    def __init__(self):
        self.lock = threading.Lock()

    def run(self, sql):
        sql = sql.strip()
        if not sql.endswith(";"):
            sql += ";"
        with self.lock:
            try:
                r = subprocess.run(
                    [os.path.abspath(BIN), "-json", DB, sql],
                    capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                return "query exceeded 900s and was aborted"
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if err and not out:
            return err
        if err:
            return out + "\n\n-- stderr --\n" + err
        return out

SESSION = Session()

PAGE = """<!doctype html><meta charset=utf-8><title>PQL console</title>
<style>
 body{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;background:#0f1115;color:#d7dae0}
 header{padding:10px 16px;background:#171a21;border-bottom:1px solid #262b36}
 header b{color:#ffd479} main{padding:16px;max-width:1100px}
 textarea{width:100%;height:150px;background:#11141a;color:#d7dae0;border:1px solid #2a3140;
   border-radius:6px;padding:10px;font:inherit;resize:vertical}
 button{margin-top:8px;padding:7px 16px;background:#2f6feb;color:#fff;border:0;border-radius:6px;
   font:inherit;cursor:pointer} button:hover{background:#4079f0}
 pre{background:#11141a;border:1px solid #2a3140;border-radius:6px;padding:12px;overflow:auto;
   white-space:pre-wrap;margin-top:14px}
 table{border-collapse:collapse;margin-top:14px;width:100%} th,td{border:1px solid #2a3140;padding:5px 9px;text-align:left}
 th{background:#1b202a;color:#ffd479} .hint{color:#7d8698;margin:10px 0}
 code{color:#9ad0a0}
</style>
<header><b>PQL</b> console &mdash; DuckDB (local build)</header>
<main>
<div class=hint>Each run is one session, so keep <code>TRAIN</code> and its <code>PREDICT</code> in the same box. Ctrl/Cmd+Enter runs.</div>
<textarea id=q>TRAIN MODEL demand PREDICT COUNT(snap_sales) FOR snapshots AT as_of HORIZON 30 DAYS OPTIONS (epochs = 60);\nPREDICT COUNT(snap_sales) FOR snapshots WHERE product_id = 133 USING MODEL demand;</textarea><br>
<button onclick=go()>Run</button>
<div id=out></div>
<script>
const el=document.getElementById('out');
async function go(){
  const sql=document.getElementById('q').value;
  el.innerHTML='<pre>running...</pre>';
  const r=await fetch('/q',{method:'POST',body:sql});
  const t=await r.text();
  let rows=null;
  try{ rows=JSON.parse(t); }catch(e){}
  if(Array.isArray(rows)&&rows.length){
    const cols=Object.keys(rows[0]);
    el.innerHTML='<table><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>'+
      rows.map(r=>'<tr>'+cols.map(c=>'<td>'+(r[c]===null?'':String(r[c]))+'</td>').join('')+'</tr>').join('')+'</table>';
  } else el.innerHTML='<pre>'+(t.trim()||'(no rows)')+'</pre>';
}
document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')go();});
</script></main>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        body = PAGE.encode()
        self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length","0"))
        sql = self.rfile.read(n).decode()
        out = SESSION.run(sql).encode()
        self.send_response(200); self.send_header("Content-Type","text/plain; charset=utf-8")
        self.send_header("Content-Length",str(len(out))); self.end_headers(); self.wfile.write(out)

if __name__ == "__main__":
    print(f"PQL console on http://127.0.0.1:{PORT}   (db: {DB})")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
