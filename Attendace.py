"""
Smart Attendance AI - Attendance system with face recognition
Only Python is needed: HTML + CSS live inside this file, charts are drawn as SVG by Python.

Install:  pip install flask opencv-contrib-python numpy
Run:      python Attendace.py
Open:     http://127.0.0.1:5000        Login: admin / admin123

Face recognition uses your computer's webcam through OpenCV (runs on the same PC as this app).
If OpenCV is not installed, everything else still works and attendance can be marked by hand.
"""
import os, io, csv, time, shutil, random, sqlite3, datetime, math
from collections import defaultdict
from flask import Flask, request, redirect, render_template_string, session, Response

try:
    import cv2
    import numpy as np
    HAS_FACE = hasattr(cv2, "face")          # needs opencv-contrib-python
except ImportError:
    cv2 = np = None
    HAS_FACE = False

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
ADMIN_USER, ADMIN_PASS = "admin", "admin123"
DB, FACES, MODEL = "attendance.db", "faces", "face_model.yml"
CAMERA_INDEX = 0          # change to 1 if you use an external webcam
MATCH_LIMIT = 70          # LBPH distance: lower = better match. Raise it if faces are not recognised.
LOW_ATTENDANCE = 75


# ---------------------------------------------------------------- database
def q(sql, *args):
    # FIX: the app calls q(sql, a, b, c) in some places and q(sql, (a, b, c)) in others.
    # If one tuple/list is passed, unpack it so SQLite gets the right number of values.
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = tuple(args[0])
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(sql, args).fetchall()
        con.commit()
        return rows
    finally:
        con.close()


def today():
    return datetime.date.today().isoformat()


NAMES = ["Aarav Sharma", "Diya Verma", "Kabir Singh", "Meera Nair", "Rohan Gupta", "Ananya Iyer",
         "Vihaan Mehta", "Isha Kapoor", "Arjun Reddy", "Saanvi Joshi", "Dev Malhotra", "Riya Bansal"]


def init_db():
    q("""CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY AUTOINCREMENT, roll TEXT, name TEXT,
         class TEXT, marks REAL, face INTEGER DEFAULT 0, joined TEXT)""")
    q("""CREATE TABLE IF NOT EXISTS attendance(id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER,
         date TEXT, time TEXT, status TEXT, method TEXT, UNIQUE(student_id, date))""")
    q("CREATE TABLE IF NOT EXISTS closed_days(date TEXT PRIMARY KEY)")
    if q("SELECT COUNT(*) c FROM students")[0]["c"]:
        return
    random.seed(5)
    days, d = [], datetime.date.today()
    while len(days) < 20:
        d -= datetime.timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    for i, name in enumerate(NAMES):
        marks = random.randint(38, 96)
        q("INSERT INTO students(roll,name,class,marks,joined) VALUES(?,?,?,?,?)",
          (f"R{101 + i}", name, "10-A", marks, days[-1].isoformat()))
        chance = min(0.98, 0.55 + marks / 100 * 0.42 + random.uniform(-.08, .05))
        for day in days:
            r = random.random()
            status = "Present" if r < chance else ("Late" if r < chance + .05 else "Absent")
            q("INSERT INTO attendance(student_id,date,time,status,method) VALUES(?,?,?,?,?)",
              (i + 1, day.isoformat(), "09:0%d" % random.randint(0, 9), status, "manual"))
    for day in days:
        q("INSERT OR IGNORE INTO closed_days(date) VALUES(?)", (day.isoformat(),))


# ---------------------------------------------------------------- analysis
def closed_days():
    return [r["date"] for r in q("SELECT date FROM closed_days ORDER BY date")]


def student_stats():
    days = closed_days()
    out = []
    for s in q("SELECT * FROM students ORDER BY roll"):
        mine = [d for d in days if d >= s["joined"]]
        present = {r["date"] for r in q(
            "SELECT date FROM attendance WHERE student_id=? AND status IN ('Present','Late')", s["id"])}
        n = len([d for d in mine if d in present])
        out.append({**dict(s), "present": n, "total": len(mine), "pct": n / len(mine) * 100 if mine else 0})
    return out


def daily_trend():
    pts = []
    for d in closed_days()[-15:]:
        total = q("SELECT COUNT(*) c FROM students WHERE joined<=?", d)[0]["c"] or 1
        pres = q("SELECT COUNT(*) c FROM attendance WHERE date=? AND status IN ('Present','Late')", d)[0]["c"]
        pts.append((d, pres / total * 100))
    return pts


def predictions():
    """Blend of the student's test average and a trend learned from all students (marks vs attendance)."""
    st = student_stats()
    pts = [(s["pct"], s["marks"]) for s in st]
    a = b = None
    if len(pts) >= 3:
        mx, my = sum(p for p, _ in pts) / len(pts), sum(m for _, m in pts) / len(pts)
        den = sum((p - mx) ** 2 for p, _ in pts)
        if den > 1e-9:
            b = sum((p - mx) * (m - my) for p, m in pts) / den
            a = my - b * mx
    for s in st:
        reg = a + b * s["pct"] if b is not None else 0.7 * s["marks"] + 0.3 * s["pct"]
        s["pred"] = max(0, min(100, 0.5 * s["marks"] + 0.5 * reg))
        s["grade"] = "A" if s["pred"] >= 80 else "B" if s["pred"] >= 65 else "C" if s["pred"] >= 50 else "D" if s["pred"] >= 40 else "F"
        if s["pct"] < LOW_ATTENDANCE or s["pred"] < 45:
            s["risk"], s["tip"] = "High", ("Attendance is below 75%. Talk to the student and parents this week."
                                          if s["pct"] < LOW_ATTENDANCE else "Needs extra revision classes and practice tests.")
        elif s["pct"] < 85 or s["pred"] < 60:
            s["risk"], s["tip"] = "Medium", "Watch closely. A few more present days will lift the result."
        else:
            s["risk"], s["tip"] = "Low", "On track. Keep attendance steady."
    return st, b


def insights(st):
    if not st or not closed_days():
        return ["No attendance recorded yet."]
    avg = sum(s["pct"] for s in st) / len(st)
    low = [s for s in st if s["pct"] < LOW_ATTENDANCE]
    best, worst = max(st, key=lambda s: s["pct"]), min(st, key=lambda s: s["pct"])
    out = [f"Overall attendance is {avg:.0f}% over {len(closed_days())} school days.",
           f"{len(low)} student(s) are below {LOW_ATTENDANCE}%." if low else "Every student is above 75%.",
           f"Best attendance: {best['name']} ({best['pct']:.0f}%). Lowest: {worst['name']} ({worst['pct']:.0f}%)."]
    wk = defaultdict(list)
    for d, v in daily_trend():
        wk[datetime.date.fromisoformat(d).strftime("%A")].append(v)
    if wk:
        name = min(wk, key=lambda k: sum(wk[k]) / len(wk[k]))
        out.append(f"{name} has the weakest turnout in recent weeks.")
    return out


# ---------------------------------------------------------------- face recognition (OpenCV)
_cascade = _rec = None


def detector():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cascade


def find_faces(gray):
    return detector().detectMultiScale(gray, 1.2, 5, minSize=(80, 80))


def crop(gray, box):
    x, y, w, h = box
    return cv2.resize(gray[y:y + h, x:x + w], (200, 200))


def train_model():
    """Learns every saved face photo. Called after registering or deleting a face."""
    global _rec
    _rec, imgs, labels = None, [], []
    if os.path.isdir(FACES):
        for sid in os.listdir(FACES):
            for f in os.listdir(os.path.join(FACES, sid)):
                img = cv2.imread(os.path.join(FACES, sid, f), 0)
                if img is not None:
                    imgs.append(img)
                    labels.append(int(sid))
    if not imgs:
        if os.path.exists(MODEL):
            os.remove(MODEL)
        return 0
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(imgs, np.array(labels))
    rec.write(MODEL)
    return len(set(labels))


def recognizer():
    global _rec
    if _rec is None and os.path.exists(MODEL):
        _rec = cv2.face.LBPHFaceRecognizer_create()
        _rec.read(MODEL)
    return _rec


def mark_present(sid, method="face"):
    now = datetime.datetime.now()
    q("INSERT OR IGNORE INTO attendance(student_id,date,time,status,method) VALUES(?,?,?,?,?)",
      (sid, today(), now.strftime("%H:%M"), "Present", method))


def video_frames():
    cam = cv2.VideoCapture(CAMERA_INDEX)
    rec = recognizer()
    names = {s["id"]: s["name"] for s in q("SELECT id,name FROM students")}
    streak = defaultdict(int)
    try:
        while cam.isOpened():
            ok, frame = cam.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for box in find_faces(gray):
                x, y, w, h = box
                label, color = "Unknown", (150, 150, 150)
                if rec is not None:
                    sid, dist = rec.predict(crop(gray, box))
                    if dist < MATCH_LIMIT and sid in names:
                        label, color = names[sid], (42, 16, 214)
                        streak[sid] += 1
                        if streak[sid] == 4:          # seen in 4 frames: mark attendance once
                            mark_present(sid)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cam.release()


# ---------------------------------------------------------------- SVG charts
RED, SOFT = "#d6102a", "#f6a9b3"


def line_svg(pts):
    if len(pts) < 2:
        return "<p class='muted'>Not enough days yet.</p>"
    W, H, p = 680, 250, 38
    xs = [p + (W - 2 * p) * i / (len(pts) - 1) for i in range(len(pts))]
    ys = [H - 36 - (H - 70) * v / 100 for _, v in pts]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    for v in (0, 50, 100):
        y = H - 36 - (H - 70) * v / 100
        s.append(f'<line x1="{p}" x2="{W - p}" y1="{y}" y2="{y}" stroke="#f1d5d9"/>'
                 f'<text x="4" y="{y + 4}" font-size="11" fill="#8a6a70">{v}%</text>')
    s.append(f'<polygon points="{xs[0]},{H - 36} {line} {xs[-1]},{H - 36}" fill="#d6102a18"/>')
    s.append(f'<polyline points="{line}" fill="none" stroke="{RED}" stroke-width="3" stroke-linejoin="round"/>')
    for i, ((d, v), x, y) in enumerate(zip(pts, xs, ys)):
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#fff" stroke="{RED}" stroke-width="2.5"/>')
        if i % 2 == 0:
            s.append(f'<text x="{x:.1f}" y="{H - 14}" text-anchor="middle" font-size="11" fill="#8a6a70">{d[5:]}</text>')
    return "".join(s) + "</svg>"


def hbar_svg(items):
    if not items:
        return "<p class='muted'>No data yet.</p>"
    W, row, lab = 640, 32, 150
    s = [f'<svg viewBox="0 0 {W} {row * len(items) + 6}" class="chart">']
    for i, (name, v) in enumerate(items):
        y, w = i * row + 4, (W - lab - 60) * v / 100
        col = RED if v < LOW_ATTENDANCE else SOFT
        s.append(f'<text x="0" y="{y + 19}" font-size="13" fill="#2a0d12">{name[:20]}</text>'
                 f'<rect x="{lab}" y="{y + 6}" width="{W - lab - 60}" height="16" rx="8" fill="#fbeaec"/>'
                 f'<rect x="{lab}" y="{y + 6}" width="{max(w, 4)}" height="16" rx="8" fill="{col}"/>'
                 f'<text x="{W - 50}" y="{y + 19}" font-size="13" font-weight="700" fill="#2a0d12">{v:.0f}%</text>')
    return "".join(s) + "</svg>"


def donut_svg(items):
    total = sum(v for _, v, _ in items)
    if not total:
        return "<p class='muted'>No data yet.</p>"
    C, cum = 2 * math.pi * 70, 0
    s = ['<svg viewBox="0 0 180 180" class="donut"><g transform="rotate(-90 90 90)">']
    for _, v, col in items:
        ln = v / total * C
        s.append(f'<circle cx="90" cy="90" r="70" fill="none" stroke="{col}" stroke-width="26" '
                 f'stroke-dasharray="{ln:.2f} {C - ln:.2f}" stroke-dashoffset="{-cum:.2f}"/>')
        cum += ln
    return "".join(s) + (f'</g><text x="90" y="88" text-anchor="middle" font-size="11" fill="#8a6a70">Records</text>'
                         f'<text x="90" y="110" text-anchor="middle" font-size="22" font-weight="800" fill="#2a0d12">{total}</text></svg>')


# ---------------------------------------------------------------- HTML + CSS
CSS = """
:root{--red:#d6102a;--red2:#9d0b1f;--soft:#fde8ea;--ink:#2a0d12;--muted:#7a5b61;--line:#f1d5d9}
*{box-sizing:border-box;margin:0}
body{font-family:'Manrope',system-ui,sans-serif;background:#fff;color:var(--ink)}
a{color:inherit}
.side{position:fixed;inset:0 auto 0 0;width:236px;background:linear-gradient(180deg,var(--red),var(--red2));color:#fff;padding:26px 16px;display:flex;flex-direction:column;gap:6px}
.logo{font-weight:800;font-size:1.2rem;padding:0 12px 22px;line-height:1.2}.logo small{display:block;font-weight:400;opacity:.8;font-size:.75rem;margin-top:4px}
.side a{text-decoration:none;padding:11px 14px;border-radius:12px;font-size:.92rem;transition:.2s}
.side a:hover{background:#ffffff22}.side a.on{background:#fff;color:var(--red);font-weight:700}
.side .out{margin-top:auto;opacity:.9}
main{margin-left:236px;padding:30px 4vw 70px;max-width:1200px}
h1{font-size:1.7rem;margin-bottom:20px}h2{font-size:1.05rem;margin-bottom:14px}
.grid{display:grid;gap:18px}.g4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}.g2{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 6px 22px #d6102a0d}
.stat small{color:var(--muted)}.stat div{font-size:2rem;font-weight:800;margin-top:4px}.stat.red{background:var(--red);color:#fff;border:0}.stat.red small{color:#ffd9de}
.red-t{color:var(--red)}.muted{color:var(--muted)}
form.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;align-items:end}
label{font-size:.78rem;color:var(--muted);display:block;margin-bottom:6px}
input,select{width:100%;padding:10px 12px;border-radius:10px;border:1px solid #e6c5ca;background:#fff;color:var(--ink);font:inherit}
input:focus,select:focus,button:focus-visible,a:focus-visible{outline:2px solid var(--red);outline-offset:2px}
button,.btn{background:var(--red);color:#fff;border:0;padding:11px 18px;border-radius:10px;font:inherit;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block;transition:.2s}
button:hover,.btn:hover{background:var(--red2)}
.btn.ghost{background:#fff;color:var(--red);border:1px solid var(--red)}.btn.ghost:hover{background:var(--soft)}.btn.sm{padding:6px 11px;font-size:.8rem}
table{width:100%;border-collapse:collapse;font-size:.9rem}.scroll{overflow-x:auto}
th{color:var(--muted);text-align:left;font-weight:600;padding:10px 8px;border-bottom:2px solid var(--line)}
td{padding:11px 8px;border-bottom:1px solid #f8e7e9;vertical-align:middle}
.tag{background:var(--soft);color:var(--red2);padding:3px 11px;border-radius:999px;font-size:.75rem;font-weight:700}
.tag.hot{background:var(--red);color:#fff}.tag.ok{background:#fff;border:1px solid var(--red);color:var(--red)}
.meter{background:var(--soft);border-radius:99px;height:8px;width:110px;display:inline-block;vertical-align:middle;margin-right:8px}.meter i{display:block;height:100%;border-radius:99px;background:var(--red)}
.chart{width:100%;height:auto}.donut{width:190px;flex:none}.flex{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.legend{display:flex;flex-direction:column;gap:8px;font-size:.9rem}.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px}
.note{padding:12px 16px;border-left:4px solid var(--red);background:var(--soft);border-radius:0 12px 12px 0;margin-bottom:16px;font-size:.92rem}
.tip{padding:11px 14px;border-left:3px solid var(--red);background:#fff7f8;border-radius:0 10px 10px 0;margin-bottom:10px;font-size:.92rem}
form.inline{display:inline}.cam{width:100%;max-width:640px;border-radius:16px;border:3px solid var(--red);background:#1a0508;display:block}
iframe{border:0;width:100%;height:320px}
.login{min-height:100vh;display:grid;grid-template-columns:1.1fr 1fr}
.login .hero{background:linear-gradient(160deg,var(--red),var(--red2));color:#fff;padding:8vw;display:flex;flex-direction:column;justify-content:center}
.login .hero h1{font-size:2.6rem;line-height:1.1}.login .hero p{margin-top:14px;opacity:.9;max-width:380px}
.login .box{display:flex;align-items:center;justify-content:center;padding:30px}.login form{width:100%;max-width:340px;display:grid;gap:14px}
@media(max-width:820px){.side{position:static;width:auto;flex-direction:row;flex-wrap:wrap;padding:14px}.logo{padding:0 10px 0 0}.side .out{margin:0}
main{margin:0;padding:20px 5vw}.login{grid-template-columns:1fr}.login .hero{padding:40px 28px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
"""
FONT = '<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">'

LAYOUT = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · Smart Attendance AI</title>""" + FONT + "<style>" + CSS + """</style></head><body>
<aside class="side"><div class="logo">Smart Attendance AI<small>Admin panel</small></div>
{% for p,n in pages %}<a href="{{ p }}" class="{{ 'on' if p==active }}">{{ n }}</a>{% endfor %}
<a class="out" href="/logout">Log out</a></aside>
<main>{% if msg %}<div class="note">{{ msg }}</div>{% endif %}{{ content|safe }}</main></body></html>"""

LOGIN = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin login</title>""" + FONT + "<style>" + CSS + """</style></head><body><div class="login">
<div class="hero"><h1>Attendance that marks itself.</h1><p>Face recognition, attendance records, analysis and performance predictions for your class.</p></div>
<div class="box"><form method="post"><h2>Admin login</h2>
{% if error %}<div class="note">{{ error }}</div>{% endif %}
<div><label>Username</label><input name="user" required autofocus></div>
<div><label>Password</label><input type="password" name="password" required></div>
<button>Log in</button></form></div></div></body></html>"""

DASH = """
<h1>Dashboard</h1>
<div class="grid g4">
 <div class="card stat red"><small>Present today</small><div>{{ present }} / {{ total }}</div></div>
 <div class="card stat"><small>Not marked today</small><div>{{ pending }}</div></div>
 <div class="card stat"><small>Overall attendance</small><div class="red-t">{{ '%.0f'|format(avg) }}%</div></div>
 <div class="card stat"><small>Faces registered</small><div>{{ faces }}</div></div>
</div><br>
<div class="grid g2">
 <div class="card"><h2>Attendance over the last school days</h2>{{ trend|safe }}</div>
 <div class="card"><h2>Quick actions</h2><div class="flex">
  <a class="btn" href="/scan">Start face scan</a><a class="btn ghost" href="/records">Attendance records</a>
  <a class="btn ghost" href="/predictions">Predictions</a></div><br>
  {% for t in tips %}<div class="tip">{{ t }}</div>{% endfor %}</div>
</div>"""

STUDENTS = """
<h1>Students</h1>
<div class="card"><h2>Add a student</h2><form class="row" method="post" action="/students/add">
 <div><label>Roll no.</label><input name="roll" required></div><div><label>Name</label><input name="name" required></div>
 <div><label>Class</label><input name="cls" value="10-A" required></div>
 <div><label>Test average (0-100)</label><input type="number" name="marks" min="0" max="100" step="0.1" value="60" required></div>
 <button>Save student</button></form></div><br>
<div class="card scroll"><table><tr><th>Roll</th><th>Name</th><th>Class</th><th>Test avg</th><th>Face</th><th>Actions</th></tr>
{% for s in students %}<tr><td>{{ s.roll }}</td><td>{{ s.name }}</td><td>{{ s.class }}</td>
<td><form class="inline" method="post" action="/students/{{ s.id }}/marks"><input style="width:80px" type="number" name="marks" min="0" max="100" step="0.1" value="{{ s.marks }}"> <button class="btn ghost sm">Update</button></form></td>
<td>{% if s.face %}<span class="tag ok">Registered</span>{% else %}<span class="tag">Not yet</span>{% endif %}</td>
<td><form class="inline" method="post" action="/students/{{ s.id }}/face"><button class="btn sm">{{ 'Re-register face' if s.face else 'Register face' }}</button></form>
<form class="inline" method="post" action="/students/{{ s.id }}/delete" onsubmit="return confirm('Delete {{ s.name }} and all attendance?')"><button class="btn ghost sm">Delete</button></form></td></tr>{% endfor %}
</table></div>
<p class="muted" style="margin-top:12px">Register face: sit in front of the webcam. The app saves about 30 photos in roughly 10 seconds. Close the Face scan page first so the camera is free.</p>"""

RECORDS = """
<h1>Attendance records</h1>
<div class="card"><form class="row" method="get"><div><label>Date</label><input type="date" name="date" value="{{ date }}"></div>
<div><label>Student</label><select name="sid"><option value="0">All students</option>{% for s in students %}<option value="{{ s.id }}" {{ 'selected' if s.id==sid }}>{{ s.name }}</option>{% endfor %}</select></div>
<button>Filter</button><a class="btn ghost" href="/records/export.csv?date={{ date }}&sid={{ sid }}">Download CSV</a></form></div><br>
<div class="grid g2">
 <div class="card"><h2>Mark by hand</h2><form class="row" method="post" action="/records/mark">
  <input type="hidden" name="date" value="{{ date }}">
  <div><label>Student</label><select name="sid">{% for s in students %}<option value="{{ s.id }}">{{ s.name }}</option>{% endfor %}</select></div>
  <div><label>Status</label><select name="status"><option>Present</option><option>Late</option><option>Absent</option></select></div><button>Save</button></form></div>
 <div class="card"><h2>Close the day</h2><p class="muted" style="margin-bottom:12px">Marks everyone without a record on {{ date }} as Absent and adds the day to the analysis.</p>
  <form method="post" action="/records/close"><input type="hidden" name="date" value="{{ date }}"><button>Close {{ date }}</button></form></div>
</div><br>
<div class="card scroll"><table><tr><th>Date</th><th>Time</th><th>Roll</th><th>Student</th><th>Status</th><th>Marked by</th></tr>
{% for r in rows %}<tr><td>{{ r.date }}</td><td>{{ r.time }}</td><td>{{ r.roll }}</td><td>{{ r.name }}</td>
<td><span class="tag {{ 'hot' if r.status=='Absent' else 'ok' }}">{{ r.status }}</span></td><td>{{ 'Face scan' if r.method=='face' else 'Admin' }}</td></tr>
{% else %}<tr><td colspan="6" class="muted">No records for this filter.</td></tr>{% endfor %}</table></div>"""

ANALYSIS = """
<h1>Attendance analysis</h1>
<div class="grid g2">
 <div class="card"><h2>Daily attendance</h2>{{ trend|safe }}</div>
 <div class="card"><h2>All records</h2><div class="flex">{{ donut|safe }}<div class="legend">
  {% for n,v,c in parts %}<div><i style="background:{{ c }}"></i>{{ n }} · {{ v }}</div>{% endfor %}</div></div><br>
  <h2>Insights</h2>{% for t in tips %}<div class="tip">{{ t }}</div>{% endfor %}</div>
</div><br>
<div class="card"><h2>Attendance by student</h2><p class="muted" style="margin-bottom:10px">Bright red bars are below {{ low }}%.</p>{{ bars|safe }}</div>"""

PREDICT = """
<h1>Student performance predictions</h1>
<div class="note">The prediction blends each student's test average with a trend learned from the whole class (how marks change with attendance). It is a guide for teachers, not a final result.</div>
<div class="card scroll"><table><tr><th>Student</th><th>Attendance</th><th>Test avg</th><th>Predicted score</th><th>Grade</th><th>Risk</th><th>Suggested action</th></tr>
{% for s in st %}<tr><td>{{ s.name }}</td><td>{{ '%.0f'|format(s.pct) }}%</td><td>{{ '%.0f'|format(s.marks) }}</td>
<td><span class="meter"><i style="width:{{ s.pred }}%"></i></span>{{ '%.0f'|format(s.pred) }}</td><td><b>{{ s.grade }}</b></td>
<td><span class="tag {{ 'hot' if s.risk=='High' else 'ok' if s.risk=='Low' else '' }}">{{ s.risk }}</span></td><td>{{ s.tip }}</td></tr>{% endfor %}</table></div>
{% if slope is not none %}<p class="muted" style="margin-top:12px">Class trend: each extra 10% attendance goes with about {{ '%.1f'|format(slope * 10) }} more marks.</p>{% endif %}"""

SCAN = """
<h1>Face recognition</h1>
{% if not has_face %}<div class="card"><h2>Face recognition is not installed</h2>
<p>Open a terminal and run <b>pip install opencv-contrib-python numpy</b>, then restart this app.</p></div>
{% elif not trained %}<div class="card"><h2>No faces registered yet</h2><p>Go to <a class="red-t" href="/students">Students</a> and press <b>Register face</b> for at least one student.</p></div>
{% else %}<div class="grid g2"><div class="card"><h2>Live camera</h2><img class="cam" src="/video_feed" alt="Live camera">
<p class="muted" style="margin-top:10px">Look at the camera. A red box with your name means attendance is marked. Leave this page to turn the camera off.</p></div>
<div class="card"><h2>Marked today</h2><iframe src="/scan/today" title="Marked today"></iframe></div></div>{% endif %}"""

TODAY = """<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="3"><style>""" + CSS + """
body{padding:4px}</style></head><body><table><tr><th>Time</th><th>Student</th><th>Marked by</th></tr>
{% for r in rows %}<tr><td>{{ r.time }}</td><td>{{ r.name }}</td><td>{{ 'Face scan' if r.method=='face' else 'Admin' }}</td></tr>
{% else %}<tr><td colspan="3" class="muted">Nobody yet.</td></tr>{% endfor %}</table></body></html>"""

PAGES = [("/", "Dashboard"), ("/students", "Students"), ("/records", "Attendance"), ("/analysis", "Analysis"),
         ("/predictions", "Predictions"), ("/scan", "Face scan")]


def render(title, active, template, **ctx):
    content = render_template_string(template, **ctx)
    return render_template_string(LAYOUT, title=title, active=active, pages=PAGES, content=content,
                                  msg=request.args.get("msg"))


# ---------------------------------------------------------------- routes
@app.before_request
def need_login():
    if request.endpoint not in ("login", "static") and not session.get("admin"):
        return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        if request.form["user"] == ADMIN_USER and request.form["password"] == ADMIN_PASS:
            session["admin"] = True
            return redirect("/")
        err = "Wrong username or password."
    return render_template_string(LOGIN, error=err)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
def dashboard():
    st = student_stats()
    present = q("SELECT COUNT(*) c FROM attendance WHERE date=? AND status IN ('Present','Late')", today())[0]["c"]
    marked = q("SELECT COUNT(*) c FROM attendance WHERE date=?", today())[0]["c"]
    return render("Dashboard", "/", DASH, present=present, total=len(st), pending=len(st) - marked,
                  avg=sum(s["pct"] for s in st) / len(st) if st else 0, faces=sum(s["face"] for s in st),
                  trend=line_svg(daily_trend()), tips=insights(st))


# ---- students
@app.route("/students")
def students():
    return render("Students", "/students", STUDENTS, students=q("SELECT * FROM students ORDER BY roll"))


@app.route("/students/add", methods=["POST"])
def add_student():
    f = request.form
    q("INSERT INTO students(roll,name,class,marks,joined) VALUES(?,?,?,?,?)",
      (f["roll"].strip(), f["name"].strip(), f["cls"].strip(), float(f["marks"]), today()))
    return redirect("/students?msg=Student added.")


@app.route("/students/<int:sid>/marks", methods=["POST"])
def update_marks(sid):
    q("UPDATE students SET marks=? WHERE id=?", float(request.form["marks"]), sid)
    return redirect("/students?msg=Test average updated.")


@app.route("/students/<int:sid>/delete", methods=["POST"])
def delete_student(sid):
    q("DELETE FROM attendance WHERE student_id=?", sid)
    q("DELETE FROM students WHERE id=?", sid)
    shutil.rmtree(os.path.join(FACES, str(sid)), ignore_errors=True)
    if HAS_FACE:
        train_model()
    return redirect("/students?msg=Student deleted.")


@app.route("/students/<int:sid>/face", methods=["POST"])
def register_face(sid):
    if not HAS_FACE:
        return redirect("/students?msg=Install opencv-contrib-python numpy to use face recognition.")
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        return redirect("/students?msg=Camera not found. Close other apps using it, or change CAMERA_INDEX.")
    folder = os.path.join(FACES, str(sid))
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    saved, start = 0, time.time()
    while saved < 30 and time.time() - start < 25:
        ok, frame = cam.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = find_faces(gray)
        if len(faces):
            box = max(faces, key=lambda b: b[2] * b[3])
            cv2.imwrite(os.path.join(folder, f"{saved}.jpg"), crop(gray, box))
            saved += 1
            time.sleep(0.2)
    cam.release()
    if saved >= 10:
        q("UPDATE students SET face=1 WHERE id=?", sid)
        train_model()
        return redirect(f"/students?msg=Face registered ({saved} photos).")
    shutil.rmtree(folder, ignore_errors=True)
    return redirect("/students?msg=No clear face seen. Face the camera in good light and try again.")


# ---- records
def record_rows(date, sid):
    sql = ("SELECT a.*, s.name, s.roll FROM attendance a JOIN students s ON s.id=a.student_id WHERE 1=1")
    args = []
    if date:
        sql, args = sql + " AND a.date=?", args + [date]
    if sid:
        sql, args = sql + " AND a.student_id=?", args + [sid]
    return q(sql + " ORDER BY a.date DESC, a.time DESC", *args)


@app.route("/records")
def records():
    date, sid = request.args.get("date") or today(), request.args.get("sid", 0, type=int)
    return render("Attendance", "/records", RECORDS, date=date, sid=sid, rows=record_rows(date, sid),
                  students=q("SELECT * FROM students ORDER BY name"))


@app.route("/records/mark", methods=["POST"])
def mark_by_hand():
    f = request.form
    q("INSERT INTO attendance(student_id,date,time,status,method) VALUES(?,?,?,?,?) "
      "ON CONFLICT(student_id,date) DO UPDATE SET status=excluded.status, method='manual'",
      f["sid"], f["date"], datetime.datetime.now().strftime("%H:%M"), f["status"], "manual")
    return redirect(f"/records?date={f['date']}&msg=Attendance saved.")


@app.route("/records/close", methods=["POST"])
def close_day():
    date = request.form["date"]
    for s in q("SELECT id FROM students WHERE joined<=?", date):
        q("INSERT OR IGNORE INTO attendance(student_id,date,time,status,method) VALUES(?,?,?,?,?)",
          s["id"], date, "--", "Absent", "manual")
    q("INSERT OR IGNORE INTO closed_days(date) VALUES(?)", date)
    return redirect(f"/records?date={date}&msg=Day closed. Absentees were marked and the day is in the analysis.")


@app.route("/records/export.csv")
def export_csv():
    rows = record_rows(request.args.get("date"), request.args.get("sid", 0, type=int))
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Date", "Time", "Roll", "Student", "Status", "Marked by"])
    for r in rows:
        w.writerow([r["date"], r["time"], r["roll"], r["name"], r["status"], r["method"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=attendance.csv"})


# ---- analysis and predictions
@app.route("/analysis")
def analysis():
    st = student_stats()
    counts = {r["status"]: r["c"] for r in q("SELECT status, COUNT(*) c FROM attendance GROUP BY status")}
    parts = [("Present", counts.get("Present", 0), RED), ("Late", counts.get("Late", 0), SOFT),
             ("Absent", counts.get("Absent", 0), "#f3dde0")]
    return render("Analysis", "/analysis", ANALYSIS, trend=line_svg(daily_trend()), donut=donut_svg(parts),
                  parts=parts, tips=insights(st), low=LOW_ATTENDANCE,
                  bars=hbar_svg(sorted([(s["name"], s["pct"]) for s in st], key=lambda x: x[1])))


@app.route("/predictions")
def predictions_page():
    st, slope = predictions()
    return render("Predictions", "/predictions", PREDICT, st=sorted(st, key=lambda s: s["pred"]), slope=slope)


# ---- face scan
@app.route("/scan")
def scan():
    return render("Face scan", "/scan", SCAN, has_face=HAS_FACE, trained=HAS_FACE and os.path.exists(MODEL))


@app.route("/video_feed")
def video_feed():
    if not HAS_FACE:
        return "Face recognition is not installed.", 503
    return Response(video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/scan/today")
def scan_today():
    rows = q("""SELECT a.time, a.method, s.name FROM attendance a JOIN students s ON s.id=a.student_id
                WHERE a.date=? AND a.status IN ('Present','Late') ORDER BY a.time DESC""", today())
    return render_template_string(TODAY, rows=rows)


if __name__ == "__main__":
    init_db()
    if HAS_FACE and os.path.isdir(FACES) and not os.path.exists(MODEL):
        train_model()
    app.run(debug=True, threaded=True)