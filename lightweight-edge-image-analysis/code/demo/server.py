"""端侧图像分析演示系统：Python 标准库 HTTP 服务 + 浏览器前端，无第三方 Web 依赖。
启动：python3 demo/server.py  然后浏览器访问 http://localhost:8000
"""
import base64, io, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import SmallCNN  # noqa

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
MODEL_PATH = os.path.join(ROOT, 'results', 'edge_int8_scripted.pt')
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

_model = None


def get_model():
    global _model
    if _model is None:
        _model = torch.jit.load(MODEL_PATH, map_location='cpu')
        _model.eval()
    return _model


def preprocess(img: Image.Image):
    img = img.convert('L').resize((28, 28), Image.BILINEAR)
    import numpy as np
    arr = np.asarray(img, dtype='float32')
    if arr.mean() > 127:          # 白底黑字自动反色，适配手写拍照
        arr = 255.0 - arr
    x = torch.from_numpy(arr / 255.0).unsqueeze(0).unsqueeze(0)
    return (x - 0.1307) / 0.3081


@torch.no_grad()
def infer(img):
    logits = get_model()(preprocess(img))[0]
    prob = F.softmax(logits, 0)
    top = torch.topk(prob, 3)
    return [{'digit': CLASSES[i], 'prob': round(float(p), 4)} for p, i in zip(top.values, top.indices)]


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>端侧图像分析演示</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f5f6f8;margin:0;color:#1a1a1a}
.wrap{max-width:760px;margin:40px auto;background:#fff;border-radius:12px;padding:28px;box-shadow:0 2px 14px rgba(0,0,0,.08)}
h1{font-size:20px;margin:0 0 6px}p.sub{color:#666;font-size:13px;margin:0 0 22px}
.row{display:flex;gap:24px;align-items:flex-start}
canvas{width:224px;height:224px;border:1px dashed #ccc;border-radius:8px;background:#000}
#out{font-size:14px;line-height:1.9}
.bar{height:8px;background:#e8eaee;border-radius:4px;overflow:hidden;width:220px}
.bar>i{display:block;height:100%;background:#2f6fed}
button{margin-top:12px;padding:8px 16px;border:0;border-radius:6px;background:#2f6fed;color:#fff;cursor:pointer}
</style></head><body><div class="wrap">
<h1>端侧图像分析演示系统</h1>
<p class="sub">CNN + 知识蒸馏 + INT8 量化 · 输入 28×28 灰度图 · 输出 Top-3 置信度</p>
<div class="row">
  <div><canvas id="cv" width="224" height="224"></canvas><br>
  <input id="f" type="file" accept="image/*"><button onclick="run()">识别</button></div>
  <div id="out">等待输入…</div>
</div></div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
cv.addEventListener('click',()=>document.getElementById('f').click());
document.getElementById('f').addEventListener('change',e=>{
  const img=new Image();img.onload=()=>{ctx.fillStyle='#000';ctx.fillRect(0,0,224,224);
    const s=Math.min(224/img.width,224/img.height);ctx.drawImage(img,(224-img.width*s)/2,(224-img.height*s)/2,img.width*s,img.height*s);};
  img.src=URL.createObjectURL(e.target.files[0]);});
async function run(){
  const c=document.createElement('canvas');c.width=28;c.height=28;
  c.getContext('2d').drawImage(cv,0,0,28,28);
  const b64=c.toDataURL('image/png').split(',')[1];
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:b64})});
  const d=await r.json();
  document.getElementById('out').innerHTML=d.top3.map((t,i)=>
    `<div>Top${i+1}：<b>${t.digit}</b>　${(t.prob*100).toFixed(1)}%<div class="bar"><i style="width:${t.prob*100}%"></i></div></div>`).join('');
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != '/predict':
            self._send(404, b'{"error":"not found"}'); return
        try:
            data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            raw = base64.b64decode(data['image'].split(',')[-1])
            img = Image.open(io.BytesIO(raw))
            top3 = infer(img)
            self._send(200, json.dumps({'top3': top3}).encode())
        except Exception as e:
            self._send(500, json.dumps({'error': str(e)}).encode())

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    print('serving on http://localhost:8000')
    HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
