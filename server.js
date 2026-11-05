import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve, extname} from 'node:path';
const root = resolve(import.meta.dirname);
const types = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'};
http.createServer(async (req,res) => {
  try {
    const path = resolve(root, '.' + decodeURIComponent(new URL(req.url,'http://localhost').pathname === '/' ? '/index.html' : new URL(req.url,'http://localhost').pathname));
    if (!path.startsWith(root + '/') || path.includes('/.')) { res.writeHead(403).end(); return; }
    const data = await readFile(path);
    res.writeHead(200, {'Content-Type':types[extname(path)] || 'application/octet-stream','Cache-Control':'no-cache'}).end(data);
  } catch { res.writeHead(404).end('Not found'); }
}).listen(Number(process.env.PORT || 3000),'0.0.0.0',()=>console.log('Breakout Lab: http://localhost:' + (process.env.PORT || 3000)));
