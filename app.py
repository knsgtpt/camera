from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from urllib.parse import quote as urlquote
import cv2, numpy as np, threading, time, os, uuid
from datetime import datetime
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'eyewatch2024'
sio = SocketIO(app, cors_allowed_origins='*', async_mode='threading', logger=False, engineio_logger=False)

BASE   = os.path.dirname(os.path.abspath(__file__))
SNAPS  = os.path.join(BASE, 'static', 'snapshots')
os.makedirs(SNAPS, exist_ok=True)

cameras = {}        # id -> config dict
monitors = {}       # id -> CameraMonitor thread
logs = []           # list of log dicts
logs_lock = threading.Lock()


class CameraMonitor(threading.Thread):
    def __init__(self, cam_id, cfg):
        super().__init__(daemon=True)
        self.cam_id  = cam_id
        self.cfg     = cfg
        self.running = True
        self.buf     = deque(maxlen=2)
        self.fgbg    = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
        self.present      = True
        self.absent_since = None
        self.snapped      = False
        self.cur_log      = None
        self.warmup       = 50
        self.frame_n      = 0
        self._last_frame  = None
        self._flock       = threading.Lock()

    def rtsp(self):
        c = self.cfg
        u = urlquote(c.get('username','admin'), safe='')
        p = urlquote(c.get('password',''),      safe='')
        return f"rtsp://{u}:{p}@{c['ip']}:{c.get('port',554)}/cam/realmonitor?channel={c.get('channel',1)}&subtype=0"

    def open_cap(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer'
        url = self.rtsp()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        if not cap.isOpened():
            cap.release(); return None
        ret, _ = cap.read()
        if not ret:
            cap.release(); return None
        return cap

    def detect(self, frame):
        small = cv2.resize(frame, (320,240))
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5,5), 0)
        mask  = self.fgbg.apply(gray)
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return cv2.countNonZero(mask) / mask.size > 0.005

    def annotate(self, frame):
        h, w = frame.shape[:2]
        ts   = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        if self.present:
            color, label = (0,200,80), '● HIEN DIEN'
        else:
            e = int((datetime.now()-self.absent_since).total_seconds())
            m, s = divmod(e, 60)
            color, label = (0,60,220), f'● VANG MAT  {m:02d}:{s:02d}'
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(w,34),(10,10,20),-1)
        cv2.addWeighted(ov,.65,frame,.35,0,frame)
        cv2.putText(frame, ts,    (8,22),  cv2.FONT_HERSHEY_SIMPLEX,.5,(180,180,180),1)
        cv2.putText(frame, label, (w-230,22), cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        cv2.putText(frame, self.cfg.get('name',''), (8,h-8), cv2.FONT_HERSHEY_SIMPLEX,.45,(160,160,255),1)
        return frame

    def snap(self, frame):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fn = f"{self.cam_id}_{ts}.jpg"
        cv2.imwrite(os.path.join(SNAPS, fn), frame)
        return fn

    def emit(self, event, **extra):
        data = dict(cam_id=self.cam_id, cam_name=self.cfg.get('name',''), event=event,
                    timestamp=datetime.now().isoformat(), **extra)
        sio.emit('ev', data)

    def run(self):
        delay = 5
        while self.running:
            cap = self.open_cap()
            if cap is None:
                cameras[self.cam_id]['status'] = 'error'
                self.emit('connect_error')
                time.sleep(delay); delay = min(delay*2, 60); continue
            cameras[self.cam_id]['status'] = 'online'
            self.emit('connected'); delay = 5
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    cameras[self.cam_id]['status'] = 'reconnecting'; break
                self.frame_n += 1
                annotated = self.annotate(frame.copy())
                with self._flock: self._last_frame = frame.copy()
                ok, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY,70])
                if ok: self.buf.append(jpg.tobytes())
                if self.frame_n < self.warmup: continue
                here = self.detect(frame)
                if self.present and not here:
                    self.present = False; self.absent_since = datetime.now(); self.snapped = False
                    entry = dict(id=str(uuid.uuid4())[:8], cam_id=self.cam_id,
                                 cam_name=self.cfg.get('name',''),
                                 absent_since=self.absent_since.isoformat(),
                                 return_time=None, duration_sec=None,
                                 snapshot=None, status='absent')
                    with logs_lock: logs.insert(0, entry)
                    self.cur_log = entry
                    self.emit('absent', log_id=entry['id'], absent_since=entry['absent_since'])
                elif not self.present and not here:
                    elapsed = (datetime.now()-self.absent_since).total_seconds()
                    if elapsed >= 300 and not self.snapped:
                        with self._flock: sf = self._last_frame.copy()
                        fn = self.snap(sf); self.snapped = True
                        if self.cur_log: self.cur_log['snapshot'] = fn
                        self.emit('snapshot', snapshot=fn)
                elif not self.present and here:
                    rt  = datetime.now()
                    dur = int((rt-self.absent_since).total_seconds())
                    self.present = True
                    if self.cur_log:
                        self.cur_log['return_time']  = rt.isoformat()
                        self.cur_log['duration_sec'] = dur
                        self.cur_log['status']       = 'returned'
                    self.emit('returned', elapsed_sec=dur)
                    self.cur_log = None
            cap.release()

    def get_frame(self):
        return self.buf[-1] if self.buf else None

    def stop(self):
        self.running = False


# ── Placeholder frame ──────────────────────────────────────────────────────
def placeholder(text='No Signal'):
    img = np.zeros((360,640,3), np.uint8); img[:] = (18,18,28)
    cv2.putText(img, text, (200,190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60,80,100), 2)
    _, jpg = cv2.imencode('.jpg', img)
    return jpg.tobytes()

PH = placeholder()


def mjpeg(cam_id):
    while True:
        m = monitors.get(cam_id)
        if m:
            f = m.get_frame()
            if f:
                yield b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + f + b'\r\n'
                time.sleep(0.04); continue
        yield b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + PH + b'\r\n'
        time.sleep(0.5)


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/static/snapshots/<p>')
def snap(p): return send_from_directory(SNAPS, p)

@app.route('/api/cameras', methods=['GET'])
def get_cams():
    return jsonify([{k:v for k,v in c.items() if k!='password'} for c in cameras.values()])

@app.route('/api/cameras', methods=['POST'])
def add_cam():
    d = request.get_json(silent=True) or {}
    if not d.get('ip'): return jsonify({'error':'IP bắt buộc'}), 400
    cid = str(uuid.uuid4())[:8]
    cameras[cid] = {
        'id': cid,
        'name':     d.get('name', f'Camera {len(cameras)+1}'),
        'ip':       d['ip'].strip(),
        'port':     int(d.get('port') or 554),
        'channel':  int(d.get('channel') or 1),
        'username': d.get('username','admin').strip(),
        'password': d.get('password',''),
        'status':   'connecting',
    }
    m = CameraMonitor(cid, cameras[cid])
    monitors[cid] = m; m.start()
    return jsonify({'id': cid, 'status': 'connecting'})

@app.route('/api/cameras/<cid>', methods=['DELETE'])
def del_cam(cid):
    if cid in monitors: monitors[cid].stop(); del monitors[cid]
    if cid in cameras:  del cameras[cid]
    return jsonify({'ok': True})

@app.route('/api/stream/<cid>')
def stream(cid):
    return Response(mjpeg(cid), mimetype='multipart/x-mixed-replace; boundary=f')

@app.route('/api/logs', methods=['GET'])
def get_logs():
    with logs_lock: return jsonify(logs[:200])

@app.route('/api/logs', methods=['DELETE'])
def clear_logs():
    with logs_lock: logs.clear()
    return jsonify({'ok': True})

@app.route('/api/test', methods=['POST'])
def test_conn():
    import socket as _sock
    d   = request.get_json(silent=True) or {}
    ip  = (d.get('ip') or '').strip()
    if not ip:
        return jsonify({'ok': False, 'msg': 'Chưa nhập địa chỉ IP', 'step': 0})

    port = int(d.get('port') or 554)

    # ── Bước 1: kiểm tra TCP port (nhanh, 3 giây) ─────────────────────────
    try:
        s = _sock.create_connection((ip, port), timeout=3)
        s.close()
    except _sock.timeout:
        return jsonify({'ok': False, 'step': 1,
                        'msg': f'[Bước 1] Timeout khi kết nối {ip}:{port} — camera không phản hồi hoặc sai IP/port'})
    except ConnectionRefusedError:
        return jsonify({'ok': False, 'step': 1,
                        'msg': f'[Bước 1] Port {port} bị từ chối — thử port 554 hoặc 8554'})
    except OSError as e:
        return jsonify({'ok': False, 'step': 1,
                        'msg': f'[Bước 1] Không đến được {ip}:{port} — {e}'})

    # ── Bước 2: kết nối RTSP qua OpenCV ───────────────────────────────────
    u   = urlquote(d.get('username', 'admin'), safe='')
    p   = urlquote(d.get('password', ''),      safe='')
    ch  = int(d.get('channel') or 1)
    url = f"rtsp://{u}:{p}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=0"

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer'
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 6000)

    if not cap.isOpened():
        cap.release()
        return jsonify({'ok': False, 'step': 2,
                        'msg': f'[Bước 2] TCP OK nhưng RTSP từ chối — sai username/password hoặc camera chưa bật RTSP'})

    # ── Bước 3: đọc frame thực ────────────────────────────────────────────
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return jsonify({'ok': False, 'step': 3,
                        'msg': '[Bước 3] RTSP mở được nhưng không đọc được hình — sai channel hoặc subtype'})

    h, w = frame.shape[:2]
    return jsonify({'ok': True, 'step': 3,
                    'msg': f'✓ Kết nối thành công! Hình {w}×{h}px từ {ip}'})

@app.route('/diag')
def diag():
    import socket as _sock
    rows = []
    for cid, cfg in cameras.items():
        m   = monitors.get(cid)
        ip  = cfg.get('ip','?')
        port= cfg.get('port', 554)
        # TCP check
        try:
            s = _sock.create_connection((ip, port), timeout=2); s.close()
            tcp = '✓ OK'
        except Exception as e:
            tcp = f'✗ {e}'
        rows.append({'id':cid,'name':cfg.get('name'),'ip':ip,'port':port,
                     'status':cfg.get('status'),'tcp':tcp,
                     'frames': len(m.buf) if m else 0})
    return jsonify({'cameras': rows,
                    'python': __import__('sys').version,
                    'cv2': cv2.__version__,
                    'total_logs': len(logs)})


@app.route('/api/demo', methods=['POST'])
def demo():
    cid = 'demo_' + str(uuid.uuid4())[:4]
    cameras[cid] = {'id':cid,'name':'Demo Camera','ip':'0.0.0.0','port':0,'channel':1,'username':'demo','status':'demo'}
    with logs_lock:
        logs.insert(0, {'id':cid,'cam_id':cid,'cam_name':'Demo Camera',
                        'absent_since':datetime.now().isoformat(),
                        'return_time':None,'duration_sec':None,'snapshot':None,'status':'absent'})
    sio.emit('ev', {'cam_id':cid,'cam_name':'Demo Camera','event':'absent','timestamp':datetime.now().isoformat()})
    return jsonify({'id':cid})


if __name__ == '__main__':
    print('='*55)
    print('  EyeWatch  →  http://localhost:5000')
    print('='*55)
    sio.run(app, host='0.0.0.0', port=5000, debug=False)
