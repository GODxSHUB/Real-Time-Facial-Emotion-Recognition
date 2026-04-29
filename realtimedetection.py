# pyright: reportMissingImports=false
"""
╔══════════════════════════════════════════════════════════════╗
║          Enhanced Real-Time Emotion Detector                 ║
╠══════════════════════════════════════════════════════════════╣
║  Visual  → HUD rounded boxes · emoji overlay · pulse ring   ║
║            live bar graph · glassmorphism sidebar            ║
║            smooth color interpolation                        ║
║  Data    → CSV logging · scrolling history sparkline         ║
║            session pie-chart · dominant-emotion timer        ║
║  Smart   → Centroid multi-face IDs · frame-avg smoother      ║
║            auto-snapshot · attention via OpenCV eye cascade  ║
║  Audio   → pygame chimes on emotion change · pyttsx3 TTS     ║
╚══════════════════════════════════════════════════════════════╝

Controls:
  q / ESC  → quit
  s        → manual snapshot
  b        → toggle bar graph
  p        → toggle glassmorphism panel
"""

import cv2
import numpy as np
import os, sys, csv, time, math, threading
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import model_from_json
# Attention detection uses OpenCV eye cascade — no mediapipe needed
import pygame
import pyttsx3
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI window)
import matplotlib.pyplot as plt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LABELS: Dict[int, str] = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise",
}

# BGR colors (OpenCV convention)
EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Angry":    (60,  20,  220),
    "Disgust":  (30,  140, 255),
    "Fear":     (180,  0,  180),
    "Happy":    (0,   210,  80),
    "Neutral":  (180, 180, 180),
    "Sad":      (220,  80,   0),
    "Surprise": (0,   220, 220),
}

EMOTION_EMOJI: Dict[str, str] = {
    "Angry": "😡", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😊", "Neutral": "😐", "Sad": "😢", "Surprise": "😲",
}

# Tone frequencies for audio chimes (Hz)
EMOTION_FREQ: Dict[str, int] = {
    "Angry": 220, "Disgust": 180, "Fear": 260,
    "Happy": 440, "Neutral": 330, "Sad": 196, "Surprise": 523,
}

SNAPSHOTS_DIR    = "snapshots"
CSV_LOG_PATH     = "emotion_log.csv"
SUMMARY_PNG      = "session_summary.png"
SNAP_CONFIDENCE  = 0.90     # auto-snapshot when confidence ≥ this
SMOOTH_N         = 8        # frames averaged for smoother predictions
HISTORY_LEN      = 150      # frames stored in sidebar sparkline
TTS_COOLDOWN     = 3.0      # min seconds between TTS announcements
MAX_DISAPPEAR    = 20       # centroid tracker patience (frames)
EAR_THRESHOLD    = 0.22     # eye-aspect-ratio below → inattentive
SIDEBAR_W        = 270      # glassmorphism panel width in pixels
CV_FONT          = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CENTROID TRACKER  — assigns persistent IDs to detected faces
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CentroidTracker:
    def __init__(self, max_disappeared: int = MAX_DISAPPEAR):
        self.next_id          = 0
        self.objects:      Dict[int, np.ndarray] = {}
        self.disappeared:  Dict[int, int]        = {}
        self.max_disappeared                     = max_disappeared

    def _register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_id]     = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid: int) -> None:
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects: list) -> Dict[int, np.ndarray]:
        """rects: list of (x, y, w, h).  Returns {face_id: centroid}."""
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        input_centroids = np.array(
            [(x + w // 2, y + h // 2) for x, y, w, h in rects], dtype="float32"
        )

        if not self.objects:
            for c in input_centroids:
                self._register(c)
        else:
            oids          = list(self.objects.keys())
            obj_centroids = np.array(list(self.objects.values()), dtype="float32")

            # Distance matrix [existing × new]
            D    = np.linalg.norm(obj_centroids[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                oid = oids[r]
                self.objects[oid]     = input_centroids[c]
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            for r in set(range(len(oids))) - used_rows:
                self.disappeared[oids[r]] += 1
                if self.disappeared[oids[r]] > self.max_disappeared:
                    self._deregister(oids[r])

            for c in set(range(len(input_centroids))) - used_cols:
                self._register(input_centroids[c])

        return self.objects


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. EMOTION SMOOTHER  — rolling average per face to kill jitter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmotionSmoother:
    def __init__(self, n: int = SMOOTH_N):
        self.n      = n
        self.buffer: Dict[int, deque] = defaultdict(lambda: deque(maxlen=n))

    def update(self, face_id: int, pred: np.ndarray) -> np.ndarray:
        self.buffer[face_id].append(pred.copy())
        return np.mean(self.buffer[face_id], axis=0)

    def cleanup(self, active_ids) -> None:
        for fid in [k for k in self.buffer if k not in active_ids]:
            del self.buffer[fid]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SESSION LOGGER  — CSV + matplotlib pie chart on exit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SessionLogger:
    def __init__(self, csv_path: str = CSV_LOG_PATH):
        self.counts: Dict[str, int] = defaultdict(int)
        self._file   = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "face_id", "emotion", "confidence"])

    def log(self, face_id: int, emotion: str, confidence: float) -> None:
        self._writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            face_id, emotion, f"{confidence:.4f}",
        ])
        self.counts[emotion] += 1

    def close(self) -> None:
        self._file.close()

    def save_pie_chart(self, path: str = SUMMARY_PNG) -> None:
        if not self.counts:
            return
        labels = list(self.counts.keys())
        sizes  = [self.counts[l] for l in labels]
        # Convert BGR → RGB for matplotlib
        colors = [tuple(v / 255 for v in EMOTION_COLORS[l][::-1]) for l in labels]

        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#12121e")
        ax.set_facecolor("#12121e")
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", colors=colors,
            startangle=140, pctdistance=0.82,
            textprops={"color": "white", "fontsize": 11},
        )
        for at in autotexts:
            at.set_fontsize(9); at.set_color("#dddddd")
        ax.set_title("Session Emotion Distribution", color="white", fontsize=14, pad=14)
        plt.tight_layout()
        plt.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Summary] Pie chart saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. AUDIO MANAGER  — sine-wave chimes + pyttsx3 TTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AudioManager:
    def __init__(self):
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        self._tts          = pyttsx3.init()
        self._tts.setProperty("rate", 155)
        self._last_tts     = 0.0
        self._last_emotion: Optional[str] = None
        self._tts_lock     = threading.Lock()
        self._sounds       = {emo: self._make_tone(freq) for emo, freq in EMOTION_FREQ.items()}

    @staticmethod
    def _make_tone(freq: int, dur: float = 0.20, vol: float = 0.30) -> pygame.mixer.Sound:
        sr   = 22050
        t    = np.linspace(0, dur, int(sr * dur), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)
        # Smooth fade-out over last 25%
        fade = int(len(wave) * 0.25)
        wave[-fade:] *= np.linspace(1.0, 0.0, fade)
        pcm  = (wave * 32767 * vol).astype(np.int16)
        stereo = np.column_stack([pcm, pcm])
        return pygame.sndarray.make_sound(stereo)

    def notify(self, emotion: str) -> None:
        if emotion == self._last_emotion:
            return
        self._last_emotion = emotion
        sound = self._sounds.get(emotion)
        if sound:
            sound.play()
        now = time.time()
        if now - self._last_tts >= TTS_COOLDOWN:
            self._last_tts = now
            threading.Thread(target=self._speak, args=(emotion,), daemon=True).start()

    def _speak(self, text: str) -> None:
        with self._tts_lock:
            self._tts.say(text)
            self._tts.runAndWait()

    def cleanup(self) -> None:
        pygame.mixer.quit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. UI RENDERER  — all drawing helpers (OpenCV + PIL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UIRenderer:
    # Attempt to load a color-emoji capable font (platform-dependent)
    _EMOJI_FONT: Optional[ImageFont.FreeTypeFont] = None
    for _fp in [
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",       # Linux
        "C:/Windows/Fonts/seguiemj.ttf",                           # Windows
        "/System/Library/Fonts/Apple Color Emoji.ttc",             # macOS
    ]:
        if os.path.exists(_fp):
            try:
                _EMOJI_FONT = ImageFont.truetype(_fp, 34)
                break
            except Exception:
                pass

    # ── Color helpers ─────────────────────────────────────────
    @staticmethod
    def lerp_color(c1, c2, t: float) -> Tuple[int, int, int]:
        """Linearly interpolate between two BGR colors."""
        return tuple(int(a + (b - a) * np.clip(t, 0, 1)) for a, b in zip(c1, c2))

    # ── Rounded rectangle ─────────────────────────────────────
    @staticmethod
    def rounded_rect(img, pt1, pt2, color, r: int = 14,
                     thickness: int = 2, filled: bool = False) -> None:
        x1, y1 = pt1; x2, y2 = pt2
        r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        lw = -1 if filled else thickness
        corners = [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]
        angles  = [(180, 270),   (270, 360),    (90, 180),    (0, 90)]
        if filled:
            cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
            cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        else:
            for (ax1, ay1), (ax2, ay2) in [
                ((x1+r, y1), (x2-r, y1)), ((x2, y1+r), (x2, y2-r)),
                ((x2-r, y2), (x1+r, y2)), ((x1, y2-r), (x1, y1+r)),
            ]:
                cv2.line(img, (ax1, ay1), (ax2, ay2), color, thickness, cv2.LINE_AA)
        for (cx, cy), (sa, ea) in zip(corners, angles):
            cv2.ellipse(img, (cx, cy), (r, r), 0, sa, ea, color, lw, cv2.LINE_AA)

    # ── Pulse / radar ring ────────────────────────────────────
    @staticmethod
    def pulse_ring(frame, cx: int, cy: int, base_r: int, tick: int, color) -> None:
        phase   = (tick % 36) / 36.0          # 0 → 1 loop
        ring_r  = int(base_r + phase * 28)
        alpha   = 1.0 - phase                  # fades as it expands
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), ring_r, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.65, frame, 1 - alpha * 0.65, 0, frame)

    # ── PIL emoji overlay ─────────────────────────────────────
    @staticmethod
    def draw_emoji(frame, emoji: str, x: int, y: int) -> None:
        if UIRenderer._EMOJI_FONT is None:
            return
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((x, y), emoji, font=UIRenderer._EMOJI_FONT, embedded_color=True)
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── Live mini bar chart ───────────────────────────────────
    @staticmethod
    def bar_graph(frame, predictions: np.ndarray, ox: int, oy: int,
                  graph_w: int = 170, graph_h: int = 145) -> None:
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (ox-10, oy-24), (ox+graph_w+10, oy+graph_h+6), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "EMOTIONS", (ox, oy-8), CV_FONT, 0.38, (160, 160, 200), 1, cv2.LINE_AA)

        row_h   = graph_h // len(LABELS)
        max_bar = graph_w - 58

        for i, (idx, name) in enumerate(LABELS.items()):
            conf  = float(predictions[idx])
            color = EMOTION_COLORS[name]
            by    = oy + i * row_h

            # Bar fill
            bw = max(int(conf * max_bar), 2)
            cv2.rectangle(frame, (ox+52, by+2), (ox+52+bw, by+row_h-2), color, -1)
            # Highlight top edge
            cv2.line(frame, (ox+52, by+2), (ox+52+bw, by+2), (255,255,255), 1)

            # Label
            cv2.putText(frame, name[:3], (ox, by+row_h-3), CV_FONT, 0.30, (180,180,200), 1, cv2.LINE_AA)
            # Percentage
            cv2.putText(frame, f"{conf*100:.0f}%", (ox+54+bw, by+row_h-3),
                        CV_FONT, 0.28, (210,210,210), 1, cv2.LINE_AA)

    # ── Glassmorphism sidebar ─────────────────────────────────
    @staticmethod
    def sidebar(frame, face_data: List[dict], history: deque,
                frame_h: int, frame_w: int) -> None:
        sw = SIDEBAR_W
        x0 = frame_w - sw

        # Dark blended background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, 0), (frame_w, frame_h), (10, 12, 22), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Left border accent
        cv2.line(frame, (x0, 0), (x0, frame_h), (70, 70, 100), 1)

        # Header
        cv2.putText(frame, "EMOTION HUD", (x0+12, 24),
                    CV_FONT, 0.52, (160, 160, 220), 1, cv2.LINE_AA)
        cv2.line(frame, (x0+6, 32), (frame_w-6, 32), (50, 50, 75), 1)

        oy = 52
        for fd in face_data[:4]:       # cap at 4 faces shown
            fid   = fd["id"]
            emo   = fd["emotion"]
            conf  = fd["confidence"]
            timer = fd["timer"]
            attn  = fd["attention"]
            col   = EMOTION_COLORS[emo]

            # Face header
            cv2.putText(frame, f"Face #{fid}", (x0+12, oy),
                        CV_FONT, 0.42, (110, 110, 155), 1, cv2.LINE_AA)
            oy += 20

            # Colored emotion badge
            UIRenderer.rounded_rect(frame, (x0+10, oy-14), (frame_w-10, oy+8),
                                    col, r=6, filled=True)
            cv2.putText(frame, f"{emo}  {conf*100:.1f}%",
                        (x0+16, oy), CV_FONT, 0.42, (15, 15, 15), 1, cv2.LINE_AA)
            oy += 20

            # Attention status
            attn_col = (0, 200, 80) if attn else (0, 80, 210)
            attn_lbl = "Attentive" if attn else "Inattentive"
            cv2.circle(frame, (x0+20, oy-4), 5, attn_col, -1, cv2.LINE_AA)
            cv2.putText(frame, attn_lbl, (x0+30, oy),
                        CV_FONT, 0.35, attn_col, 1, cv2.LINE_AA)
            oy += 18

            # Hold timer
            cv2.putText(frame, f"Held: {timer:.1f}s", (x0+14, oy),
                        CV_FONT, 0.34, (150, 150, 200), 1, cv2.LINE_AA)
            oy += 18

            cv2.line(frame, (x0+6, oy), (frame_w-6, oy), (35, 35, 55), 1)
            oy += 12

        # ── Scrolling emotion sparkline ──────────────────────
        if len(history) > 1:
            cv2.putText(frame, "History", (x0+12, oy+14),
                        CV_FONT, 0.38, (140, 140, 200), 1, cv2.LINE_AA)
            oy += 24
            sh    = 58
            sw2   = sw - 24
            xstep = sw2 / max(len(history) - 1, 1)
            pts   = []
            for i, (emo_h, _) in enumerate(history):
                emo_idx = list(LABELS.values()).index(emo_h)
                px = int(x0 + 12 + i * xstep)
                py = int(oy + sh - (emo_idx / 6.0) * sh)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (100, 200, 255), 1, cv2.LINE_AA)
            # Y-axis emotion labels
            for idx, name in LABELS.items():
                py = int(oy + sh - (idx / 6.0) * sh)
                cv2.putText(frame, name[:3], (x0 + 14, py + 3),
                            CV_FONT, 0.25, (80, 80, 110), 1, cv2.LINE_AA)

    # ── Dominant emotion timer label ──────────────────────────
    @staticmethod
    def hold_timer(frame, x: int, y: int, emotion: str, seconds: float, color) -> None:
        label = f"{seconds:.1f}s"
        cv2.putText(frame, label, (x, y), CV_FONT, 0.36, color, 1, cv2.LINE_AA)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. MODEL + PREPROCESSING UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model(json_path: str, weights_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Architecture file not found: {json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    with open(json_path, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    print(f"[Model] Loaded from {json_path} + {weights_path}")
    return model

def preprocess(gray_face: np.ndarray) -> np.ndarray:
    img = cv2.resize(gray_face, (48, 48))
    return img.reshape(1, 48, 48, 1).astype("float32") / 255.0

def detect_eyes(eye_cascade, gray_face: np.ndarray) -> int:
    """Return number of eyes detected (0-2) inside a grayscale face crop."""
    eyes = eye_cascade.detectMultiScale(
        gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    return min(len(eyes), 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. MAIN DETECTION LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(model, camera_index: int = 0) -> None:
    # ── Init OpenCV haar cascade ──────────────────────────────
    haar_path    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade classifier.")

    # ── Init camera ───────────────────────────────────────────
    webcam = cv2.VideoCapture(camera_index)
    if not webcam.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    # ── Init subsystems ───────────────────────────────────────
    tracker  = CentroidTracker()
    smoother = EmotionSmoother()
    logger   = SessionLogger()
    audio    = AudioManager()
    renderer = UIRenderer()

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    if eye_cascade.empty():
        print("[Warning] Eye cascade not found — attention detection disabled.")

    # ── Per-face persistent state ─────────────────────────────
    dom_emo:   Dict[int, str]   = {}   # current dominant emotion per face
    dom_start: Dict[int, float] = {}   # when that emotion started
    prev_col:  Dict[int, tuple] = {}   # previous interpolated color
    last_snap: Dict[int, str]   = {}   # last auto-snapped emotion per face

    history: deque = deque(maxlen=HISTORY_LEN)   # (emotion, confidence) tuples

    # ── UI toggle state ───────────────────────────────────────
    show_bars    = True
    show_sidebar = True

    # ── FPS tracking ──────────────────────────────────────────
    fps_val   = 0.0
    fps_timer = time.time()
    fps_count = 0

    tick = 0
    print("Running  —  q/ESC to quit · s = manual snapshot · b = bar graph · p = panel")

    try:
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("[Warning] Frame grab failed — exiting.")
                break

            frame_h, frame_w = frame.shape[:2]
            tick      += 1
            fps_count += 1
            if fps_count == 20:
                fps_val   = 20.0 / (time.time() - fps_timer)
                fps_timer = time.time()
                fps_count = 0

            # ── Detect faces (haar) ───────────────────────────
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
            )
            rects_list = [tuple(r) for r in rects] if len(rects) else []

            # ── Update centroid tracker ───────────────────────
            id_map = tracker.update(rects_list)
            smoother.cleanup(set(id_map.keys()))

            # ── OpenCV eye cascade attention ──────────────────
            ear_map: Dict[int, int] = {}   # face_id → eye count (0/1/2)

            # ── Per-face processing ───────────────────────────
            face_data: List[dict] = []

            for rect_idx, (x, y, w_r, h_r) in enumerate(rects_list):
                cx, cy = x + w_r // 2, y + h_r // 2

                # Find best matching face ID
                if id_map:
                    face_id = min(
                        id_map,
                        key=lambda k: float(np.linalg.norm(id_map[k] - np.array([cx, cy])))
                    )
                else:
                    face_id = 0

                # ── Predict & smooth ──────────────────────────
                face_gray   = gray[y: y+h_r, x: x+w_r]
                raw_pred    = model.predict(preprocess(face_gray), verbose=0)[0]
                smooth_pred = smoother.update(face_id, raw_pred)

                label_idx  = int(np.argmax(smooth_pred))
                emotion    = LABELS[label_idx]
                confidence = float(smooth_pred[label_idx])

                # ── Dominant emotion timer ────────────────────
                if dom_emo.get(face_id) != emotion:
                    dom_emo[face_id]   = emotion
                    dom_start[face_id] = time.time()
                hold_secs = time.time() - dom_start.get(face_id, time.time())

                # ── Attention (eye cascade) ──────────────────
                face_roi  = gray[y: y+h_r, x: x+w_r]
                eye_count = detect_eyes(eye_cascade, face_roi)
                ear_map[face_id] = eye_count
                attentive = eye_count >= 2

                # ── Audio feedback ────────────────────────────
                audio.notify(emotion)

                # ── CSV logging ───────────────────────────────
                logger.log(face_id, emotion, confidence)

                # ── Auto-snapshot ─────────────────────────────
                if confidence >= SNAP_CONFIDENCE and last_snap.get(face_id) != emotion:
                    last_snap[face_id] = emotion
                    fname = os.path.join(
                        SNAPSHOTS_DIR,
                        f"{emotion}_{datetime.now().strftime('%H%M%S')}_f{face_id}.jpg"
                    )
                    cv2.imwrite(fname, frame)
                    print(f"[Snapshot] {fname}")

                # ── History ───────────────────────────────────
                history.append((emotion, confidence))

                # ── Color interpolation ───────────────────────
                target = EMOTION_COLORS[emotion]
                prev   = prev_col.get(face_id, target)
                color  = UIRenderer.lerp_color(prev, target, 0.20)
                prev_col[face_id] = color

                # ── Draw: pulse ring ──────────────────────────
                UIRenderer.pulse_ring(frame, cx, cy, max(w_r, h_r) // 2 + 8,
                                      tick + face_id * 11, color)

                # ── Draw: rounded face bounding box ───────────
                UIRenderer.rounded_rect(frame, (x, y), (x+w_r, y+h_r),
                                        color, r=14, thickness=2)

                # ── Draw: emoji (PIL) ─────────────────────────
                emoji = EMOTION_EMOJI.get(emotion, "")
                UIRenderer.draw_emoji(frame, emoji, x + w_r//2 - 17, max(y - 46, 2))

                # ── Draw: label badge ─────────────────────────
                badge_txt = f"#{face_id} {emotion}  {confidence*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(badge_txt, CV_FONT, 0.46, 1)
                badge_y = max(y - 8, th + 8)
                UIRenderer.rounded_rect(frame,
                    (x, badge_y - th - 6), (x + tw + 10, badge_y + 3),
                    color, r=6, filled=True)
                cv2.putText(frame, badge_txt, (x+5, badge_y - 2),
                            CV_FONT, 0.46, (10, 10, 10), 1, cv2.LINE_AA)

                # ── Draw: attention dot ───────────────────────
                dot_col = (0, 210, 80) if attentive else (0, 70, 220)
                cv2.circle(frame, (x + w_r - 10, y + 10), 7, dot_col, -1, cv2.LINE_AA)
                cv2.circle(frame, (x + w_r - 10, y + 10), 7, (255,255,255), 1, cv2.LINE_AA)

                # ── Draw: hold timer below box ────────────────
                UIRenderer.hold_timer(frame, x+2, y+h_r+16, emotion, hold_secs, color)

                # ── Draw: eye count (small, below box) ───────
                eye_count = ear_map.get(face_id, 0)
                cv2.putText(frame, f"Eyes:{eye_count}/2", (x+2, y+h_r+30),
                            CV_FONT, 0.28, (130, 130, 160), 1, cv2.LINE_AA)

                face_data.append({
                    "id": face_id, "emotion": emotion,
                    "confidence": confidence, "timer": hold_secs,
                    "attention": attentive,
                })

                # ── Live bar graph (first face, bottom-left) ──
                if show_bars and rect_idx == 0:
                    UIRenderer.bar_graph(frame, smooth_pred,
                                         ox=10, oy=frame_h - 168)

            # ── Glassmorphism sidebar ─────────────────────────
            if show_sidebar:
                UIRenderer.sidebar(frame, face_data, history, frame_h, frame_w)

            # ── FPS + face count overlay ──────────────────────
            info_txt = f"FPS:{fps_val:.1f}  Faces:{len(rects_list)}"
            cv2.putText(frame, info_txt, (10, 22),
                        CV_FONT, 0.46, (160, 220, 160), 1, cv2.LINE_AA)

            cv2.imshow("Emotion Detector  [q/ESC quit | s snapshot | b bars | p panel]", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):           # quit
                break
            elif key == ord("s"):               # manual snapshot
                manual_path = os.path.join(SNAPSHOTS_DIR, f"manual_{datetime.now().strftime('%H%M%S%f')}.jpg")
                cv2.imwrite(manual_path, frame)
                print(f"[Manual Snapshot] {manual_path}")
            elif key == ord("b"):               # toggle bar graph
                show_bars = not show_bars
            elif key == ord("p"):               # toggle sidebar panel
                show_sidebar = not show_sidebar

    finally:
        webcam.release()
        cv2.destroyAllWindows()
        logger.close()
        logger.save_pie_chart()
        audio.cleanup()
        print("\n[Done] CSV log and session pie chart saved. Goodbye!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    try:
        model = load_model("emotiondetector.json", "emotiondetector.h5")
        run(model, camera_index=1)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)














































# # pyright: reportMissingImports=false
# """
# ╔══════════════════════════════════════════════════════════════╗
# ║          Enhanced Real-Time Emotion Detector                 ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  Visual  → HUD rounded boxes · emoji overlay · pulse ring   ║
# ║            live bar graph · glassmorphism sidebar            ║
# ║            smooth color interpolation                        ║
# ║  Data    → CSV logging · scrolling history sparkline         ║
# ║            session pie-chart · dominant-emotion timer        ║
# ║  Smart   → Centroid multi-face IDs · frame-avg smoother      ║
# ║            auto-snapshot · attention via OpenCV eye cascade  ║
# ║  Audio   → pygame chimes on emotion change · pyttsx3 TTS     ║
# ╚══════════════════════════════════════════════════════════════╝

# Controls:
#   q / ESC  → quit
#   s        → manual snapshot
#   b        → toggle bar graph
#   p        → toggle glassmorphism panel
# """

# import cv2
# import numpy as np
# import os, sys, csv, time, math, threading
# from collections import deque, defaultdict
# from datetime import datetime
# from typing import Dict, List, Tuple, Optional

# from PIL import Image, ImageDraw, ImageFont
# from tensorflow.keras.models import model_from_json
# # Attention detection uses OpenCV eye cascade — no mediapipe needed
# import pygame
# import pyttsx3
# import matplotlib
# matplotlib.use("Agg") 
# import matplotlib.pyplot as plt


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # CONFIGURATION
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LABELS: Dict[int, str] = {
#     0: "Angry", 1: "Disgust", 2: "Fear",
#     3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise",
# }

# # BGR colors (OpenCV convention)
# EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
#     "Angry":    (60,  20,  220),
#     "Disgust":  (30,  140, 255),
#     "Fear":     (180,  0,  180),
#     "Happy":    (0,   210,  80),
#     "Neutral":  (180, 180, 180),
#     "Sad":      (220,  80,   0),
#     "Surprise": (0,   220, 220),
# }

# EMOTION_EMOJI: Dict[str, str] = {
#     "Angry": "😡", "Disgust": "🤢", "Fear": "😨",
#     "Happy": "😊", "Neutral": "😐", "Sad": "😢", "Surprise": "😲",
# }

# # Tone frequencies for audio chimes (Hz)
# EMOTION_FREQ: Dict[str, int] = {
#     "Angry": 220, "Disgust": 180, "Fear": 260,
#     "Happy": 440, "Neutral": 330, "Sad": 196, "Surprise": 523,
# }

# SNAPSHOTS_DIR    = "snapshots"
# CSV_LOG_PATH     = "emotion_log.csv"
# SUMMARY_PNG      = "session_summary.png"
# SNAP_CONFIDENCE  = 0.90     # auto-snapshot when confidence ≥ this
# SMOOTH_N         = 8        # frames averaged for smoother predictions
# HISTORY_LEN      = 150      # frames stored in sidebar sparkline
# TTS_COOLDOWN     = 3.0      # min seconds between TTS announcements
# MAX_DISAPPEAR    = 20       # centroid tracker patience (frames)
# EAR_THRESHOLD    = 0.22     # eye-aspect-ratio below → inattentive
# SIDEBAR_W        = 270      # glassmorphism panel width in pixels
# CV_FONT          = cv2.FONT_HERSHEY_SIMPLEX

# os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 1. CENTROID TRACKER  — assigns persistent IDs to detected faces
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# class CentroidTracker:
#     def __init__(self, max_disappeared: int = MAX_DISAPPEAR):
#         self.next_id          = 0
#         self.objects:      Dict[int, np.ndarray] = {}
#         self.disappeared:  Dict[int, int]        = {}
#         self.max_disappeared                     = max_disappeared

#     def _register(self, centroid: np.ndarray) -> None:
#         self.objects[self.next_id]     = centroid
#         self.disappeared[self.next_id] = 0
#         self.next_id += 1

#     def _deregister(self, oid: int) -> None:
#         self.objects.pop(oid, None)
#         self.disappeared.pop(oid, None)

#     def update(self, rects: list) -> Dict[int, np.ndarray]:
#         """rects: list of (x, y, w, h).  Returns {face_id: centroid}."""
#         if not rects:
#             for oid in list(self.disappeared):
#                 self.disappeared[oid] += 1
#                 if self.disappeared[oid] > self.max_disappeared:
#                     self._deregister(oid)
#             return self.objects

#         input_centroids = np.array(
#             [(x + w // 2, y + h // 2) for x, y, w, h in rects], dtype="float32"
#         )

#         if not self.objects:
#             for c in input_centroids:
#                 self._register(c)
#         else:
#             oids          = list(self.objects.keys())
#             obj_centroids = np.array(list(self.objects.values()), dtype="float32")

#             # Distance matrix [existing × new]
#             D    = np.linalg.norm(obj_centroids[:, None] - input_centroids[None, :], axis=2)
#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]

#             used_rows, used_cols = set(), set()
#             for r, c in zip(rows, cols):
#                 if r in used_rows or c in used_cols:
#                     continue
#                 oid = oids[r]
#                 self.objects[oid]     = input_centroids[c]
#                 self.disappeared[oid] = 0
#                 used_rows.add(r)
#                 used_cols.add(c)

#             for r in set(range(len(oids))) - used_rows:
#                 self.disappeared[oids[r]] += 1
#                 if self.disappeared[oids[r]] > self.max_disappeared:
#                     self._deregister(oids[r])

#             for c in set(range(len(input_centroids))) - used_cols:
#                 self._register(input_centroids[c])

#         return self.objects


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 2. EMOTION SMOOTHER  — rolling average per face to kill jitter
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# class EmotionSmoother:
#     def __init__(self, n: int = SMOOTH_N):
#         self.n      = n
#         self.buffer: Dict[int, deque] = defaultdict(lambda: deque(maxlen=n))

#     def update(self, face_id: int, pred: np.ndarray) -> np.ndarray:
#         self.buffer[face_id].append(pred.copy())
#         return np.mean(self.buffer[face_id], axis=0)

#     def cleanup(self, active_ids) -> None:
#         for fid in [k for k in self.buffer if k not in active_ids]:
#             del self.buffer[fid]


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 3. SESSION LOGGER  — CSV + matplotlib pie chart on exit
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# class SessionLogger:
#     def __init__(self, csv_path: str = CSV_LOG_PATH):
#         self.counts: Dict[str, int] = defaultdict(int)
#         self._file   = open(csv_path, "w", newline="", encoding="utf-8")
#         self._writer = csv.writer(self._file)
#         self._writer.writerow(["timestamp", "face_id", "emotion", "confidence"])

#     def log(self, face_id: int, emotion: str, confidence: float) -> None:
#         self._writer.writerow([
#             datetime.now().isoformat(timespec="milliseconds"),
#             face_id, emotion, f"{confidence:.4f}",
#         ])
#         self.counts[emotion] += 1

#     def close(self) -> None:
#         self._file.close()

#     def save_pie_chart(self, path: str = SUMMARY_PNG) -> None:
#         if not self.counts:
#             return
#         labels = list(self.counts.keys())
#         sizes  = [self.counts[l] for l in labels]
#         # Convert BGR → RGB for matplotlib
#         colors = [tuple(v / 255 for v in EMOTION_COLORS[l][::-1]) for l in labels]

#         fig, ax = plt.subplots(figsize=(6, 5), facecolor="#12121e")
#         ax.set_facecolor("#12121e")
#         wedges, texts, autotexts = ax.pie(
#             sizes, labels=labels, autopct="%1.1f%%", colors=colors,
#             startangle=140, pctdistance=0.82,
#             textprops={"color": "white", "fontsize": 11},
#         )
#         for at in autotexts:
#             at.set_fontsize(9); at.set_color("#dddddd")
#         ax.set_title("Session Emotion Distribution", color="white", fontsize=14, pad=14)
#         plt.tight_layout()
#         plt.savefig(path, dpi=130, facecolor=fig.get_facecolor())
#         plt.close()
#         print(f"[Summary] Pie chart saved → {path}")


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 4. AUDIO MANAGER  — sine-wave chimes + pyttsx3 TTS
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# class AudioManager:
#     def __init__(self):
#         pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
#         pygame.mixer.init()
#         self._tts          = pyttsx3.init()
#         self._tts.setProperty("rate", 155)
#         self._last_tts     = 0.0
#         self._last_emotion: Optional[str] = None
#         self._tts_lock     = threading.Lock()
#         self._sounds       = {emo: self._make_tone(freq) for emo, freq in EMOTION_FREQ.items()}

#     @staticmethod
#     def _make_tone(freq: int, dur: float = 0.20, vol: float = 0.30) -> pygame.mixer.Sound:
#         sr   = 22050
#         t    = np.linspace(0, dur, int(sr * dur), endpoint=False)
#         wave = np.sin(2 * np.pi * freq * t)
#         # Smooth fade-out over last 25%
#         fade = int(len(wave) * 0.25)
#         wave[-fade:] *= np.linspace(1.0, 0.0, fade)
#         pcm  = (wave * 32767 * vol).astype(np.int16)
#         stereo = np.column_stack([pcm, pcm])
#         return pygame.sndarray.make_sound(stereo)

#     def notify(self, emotion: str) -> None:
#         if emotion == self._last_emotion:
#             return
#         self._last_emotion = emotion
#         sound = self._sounds.get(emotion)
#         if sound:
#             sound.play()
#         now = time.time()
#         if now - self._last_tts >= TTS_COOLDOWN:
#             self._last_tts = now
#             threading.Thread(target=self._speak, args=(emotion,), daemon=True).start()

#     def _speak(self, text: str) -> None:
#         with self._tts_lock:
#             self._tts.say(text)
#             self._tts.runAndWait()

#     def cleanup(self) -> None:
#         pygame.mixer.quit()


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 5. UI RENDERER  — all drawing helpers (OpenCV + PIL)
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# class UIRenderer:
#     # Attempt to load a color-emoji capable font (platform-dependent)
#     _EMOJI_FONT: Optional[ImageFont.FreeTypeFont] = None
#     for _fp in [
#         "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",       # Linux
#         "C:/Windows/Fonts/seguiemj.ttf",                           # Windows
#         "/System/Library/Fonts/Apple Color Emoji.ttc",             # macOS
#     ]:
#         if os.path.exists(_fp):
#             try:
#                 _EMOJI_FONT = ImageFont.truetype(_fp, 34)
#                 break
#             except Exception:
#                 pass

#     # ── Color helpers ─────────────────────────────────────────
#     @staticmethod
#     def lerp_color(c1, c2, t: float) -> Tuple[int, int, int]:
#         """Linearly interpolate between two BGR colors."""
#         return tuple(int(a + (b - a) * np.clip(t, 0, 1)) for a, b in zip(c1, c2))

#     # ── Rounded rectangle ─────────────────────────────────────
#     @staticmethod
#     def rounded_rect(img, pt1, pt2, color, r: int = 14,
#                      thickness: int = 2, filled: bool = False) -> None:
#         x1, y1 = pt1; x2, y2 = pt2
#         r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
#         lw = -1 if filled else thickness
#         corners = [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]
#         angles  = [(180, 270),   (270, 360),    (90, 180),    (0, 90)]
#         if filled:
#             cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
#             cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
#         else:
#             for (ax1, ay1), (ax2, ay2) in [
#                 ((x1+r, y1), (x2-r, y1)), ((x2, y1+r), (x2, y2-r)),
#                 ((x2-r, y2), (x1+r, y2)), ((x1, y2-r), (x1, y1+r)),
#             ]:
#                 cv2.line(img, (ax1, ay1), (ax2, ay2), color, thickness, cv2.LINE_AA)
#         for (cx, cy), (sa, ea) in zip(corners, angles):
#             cv2.ellipse(img, (cx, cy), (r, r), 0, sa, ea, color, lw, cv2.LINE_AA)

#     # ── Pulse / radar ring ────────────────────────────────────
#     @staticmethod
#     def pulse_ring(frame, cx: int, cy: int, base_r: int, tick: int, color) -> None:
#         phase   = (tick % 36) / 36.0          # 0 → 1 loop
#         ring_r  = int(base_r + phase * 28)
#         alpha   = 1.0 - phase                  # fades as it expands
#         overlay = frame.copy()
#         cv2.circle(overlay, (cx, cy), ring_r, color, 2, cv2.LINE_AA)
#         cv2.addWeighted(overlay, alpha * 0.65, frame, 1 - alpha * 0.65, 0, frame)

#     # ── PIL emoji overlay ─────────────────────────────────────
#     @staticmethod
#     def draw_emoji(frame, emoji: str, x: int, y: int) -> None:
#         if UIRenderer._EMOJI_FONT is None:
#             return
#         pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         ImageDraw.Draw(pil).text((x, y), emoji, font=UIRenderer._EMOJI_FONT, embedded_color=True)
#         frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

#     # ── Live mini bar chart ───────────────────────────────────
#     @staticmethod
#     def bar_graph(frame, predictions: np.ndarray, ox: int, oy: int,
#                   graph_w: int = 170, graph_h: int = 145) -> None:
#         # Semi-transparent background
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (ox-10, oy-24), (ox+graph_w+10, oy+graph_h+6), (10, 10, 15), -1)
#         cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

#         cv2.putText(frame, "EMOTIONS", (ox, oy-8), CV_FONT, 0.38, (160, 160, 200), 1, cv2.LINE_AA)

#         row_h   = graph_h // len(LABELS)
#         max_bar = graph_w - 58

#         for i, (idx, name) in enumerate(LABELS.items()):
#             conf  = float(predictions[idx])
#             color = EMOTION_COLORS[name]
#             by    = oy + i * row_h

#             # Bar fill
#             bw = max(int(conf * max_bar), 2)
#             cv2.rectangle(frame, (ox+52, by+2), (ox+52+bw, by+row_h-2), color, -1)
#             # Highlight top edge
#             cv2.line(frame, (ox+52, by+2), (ox+52+bw, by+2), (255,255,255), 1)

#             # Label
#             cv2.putText(frame, name[:3], (ox, by+row_h-3), CV_FONT, 0.30, (180,180,200), 1, cv2.LINE_AA)
#             # Percentage
#             cv2.putText(frame, f"{conf*100:.0f}%", (ox+54+bw, by+row_h-3),
#                         CV_FONT, 0.28, (210,210,210), 1, cv2.LINE_AA)

#     # ── Glassmorphism sidebar ─────────────────────────────────
#     @staticmethod
#     def sidebar(frame, face_data: List[dict], history: deque,
#                 frame_h: int, frame_w: int) -> None:
#         sw = SIDEBAR_W
#         x0 = frame_w - sw

#         # Dark blended background
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (x0, 0), (frame_w, frame_h), (10, 12, 22), -1)
#         cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#         # Left border accent
#         cv2.line(frame, (x0, 0), (x0, frame_h), (70, 70, 100), 1)

#         # Header
#         cv2.putText(frame, "EMOTION HUD", (x0+12, 24),
#                     CV_FONT, 0.52, (160, 160, 220), 1, cv2.LINE_AA)
#         cv2.line(frame, (x0+6, 32), (frame_w-6, 32), (50, 50, 75), 1)

#         oy = 52
#         for fd in face_data[:4]:       # cap at 4 faces shown
#             fid   = fd["id"]
#             emo   = fd["emotion"]
#             conf  = fd["confidence"]
#             timer = fd["timer"]
#             attn  = fd["attention"]
#             col   = EMOTION_COLORS[emo]

#             # Face header
#             cv2.putText(frame, f"Face #{fid}", (x0+12, oy),
#                         CV_FONT, 0.42, (110, 110, 155), 1, cv2.LINE_AA)
#             oy += 20

#             # Colored emotion badge
#             UIRenderer.rounded_rect(frame, (x0+10, oy-14), (frame_w-10, oy+8),
#                                     col, r=6, filled=True)
#             cv2.putText(frame, f"{emo}  {conf*100:.1f}%",
#                         (x0+16, oy), CV_FONT, 0.42, (15, 15, 15), 1, cv2.LINE_AA)
#             oy += 20

#             # Attention status
#             attn_col = (0, 200, 80) if attn else (0, 80, 210)
#             attn_lbl = "Attentive" if attn else "Inattentive"
#             cv2.circle(frame, (x0+20, oy-4), 5, attn_col, -1, cv2.LINE_AA)
#             cv2.putText(frame, attn_lbl, (x0+30, oy),
#                         CV_FONT, 0.35, attn_col, 1, cv2.LINE_AA)
#             oy += 18

#             # Hold timer
#             cv2.putText(frame, f"Held: {timer:.1f}s", (x0+14, oy),
#                         CV_FONT, 0.34, (150, 150, 200), 1, cv2.LINE_AA)
#             oy += 18

#             cv2.line(frame, (x0+6, oy), (frame_w-6, oy), (35, 35, 55), 1)
#             oy += 12

#         # ── Scrolling emotion sparkline ──────────────────────
#         if len(history) > 1:
#             cv2.putText(frame, "History", (x0+12, oy+14),
#                         CV_FONT, 0.38, (140, 140, 200), 1, cv2.LINE_AA)
#             oy += 24
#             sh    = 58
#             sw2   = sw - 24
#             xstep = sw2 / max(len(history) - 1, 1)
#             pts   = []
#             for i, (emo_h, _) in enumerate(history):
#                 emo_idx = list(LABELS.values()).index(emo_h)
#                 px = int(x0 + 12 + i * xstep)
#                 py = int(oy + sh - (emo_idx / 6.0) * sh)
#                 pts.append((px, py))
#             for i in range(1, len(pts)):
#                 cv2.line(frame, pts[i-1], pts[i], (100, 200, 255), 1, cv2.LINE_AA)
#             # Y-axis emotion labels
#             for idx, name in LABELS.items():
#                 py = int(oy + sh - (idx / 6.0) * sh)
#                 cv2.putText(frame, name[:3], (x0 + 14, py + 3),
#                             CV_FONT, 0.25, (80, 80, 110), 1, cv2.LINE_AA)

#     # ── Dominant emotion timer label ──────────────────────────
#     @staticmethod
#     def hold_timer(frame, x: int, y: int, emotion: str, seconds: float, color) -> None:
#         label = f"{seconds:.1f}s"
#         cv2.putText(frame, label, (x, y), CV_FONT, 0.36, color, 1, cv2.LINE_AA)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 6. MODEL + PREPROCESSING UTILITIES
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# def load_model(json_path: str, weights_path: str):
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"Architecture file not found: {json_path}")
#     if not os.path.exists(weights_path):
#         raise FileNotFoundError(f"Weights file not found: {weights_path}")
#     with open(json_path, "r") as f:
#         model = model_from_json(f.read())
#     model.load_weights(weights_path)
#     print(f"[Model] Loaded from {json_path} + {weights_path}")
#     return model

# def preprocess(gray_face: np.ndarray) -> np.ndarray:
#     img = cv2.resize(gray_face, (48, 48))
#     return img.reshape(1, 48, 48, 1).astype("float32") / 255.0

# def detect_eyes(eye_cascade, gray_face: np.ndarray) -> int:
#     """Return number of eyes detected (0-2) inside a grayscale face crop."""
#     eyes = eye_cascade.detectMultiScale(
#         gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
#     )
#     return min(len(eyes), 2)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 7. MAIN DETECTION LOOP
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# def run(model, camera_source=0) -> None:
#     # ── Init OpenCV haar cascade ──────────────────────────────
#     haar_path    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     face_cascade = cv2.CascadeClassifier(haar_path)
#     if face_cascade.empty():
#         raise RuntimeError("Failed to load Haar cascade classifier.")

#     # ── Init camera ───────────────────────────────────────────
#     webcam = cv2.VideoCapture(camera_source)
#     if not webcam.isOpened():
#         raise RuntimeError(f"Cannot open camera at source {camera_source}.")

#     # ── Init subsystems ───────────────────────────────────────
#     tracker  = CentroidTracker()
#     smoother = EmotionSmoother()
#     logger   = SessionLogger()
#     audio    = AudioManager()
#     renderer = UIRenderer()

#     eye_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_eye.xml"
#     )
#     if eye_cascade.empty():
#         print("[Warning] Eye cascade not found — attention detection disabled.")

#     # ── Per-face persistent state ─────────────────────────────
#     dom_emo:   Dict[int, str]   = {}   # current dominant emotion per face
#     dom_start: Dict[int, float] = {}   # when that emotion started
#     prev_col:  Dict[int, tuple] = {}   # previous interpolated color
#     last_snap: Dict[int, str]   = {}   # last auto-snapped emotion per face

#     history: deque = deque(maxlen=HISTORY_LEN)   # (emotion, confidence) tuples

#     # ── UI toggle state ───────────────────────────────────────
#     show_bars    = True
#     show_sidebar = True

#     # ── FPS tracking ──────────────────────────────────────────
#     fps_val   = 0.0
#     fps_timer = time.time()
#     fps_count = 0

#     tick = 0
#     print("Running  —  q/ESC to quit · s = manual snapshot · b = bar graph · p = panel")

#     try:
#         while True:
#             ret, frame = webcam.read()
#             if not ret:
#                 print("[Warning] Frame grab failed — exiting.")
#                 break

#             frame_h, frame_w = frame.shape[:2]
#             tick      += 1
#             fps_count += 1
#             if fps_count == 20:
#                 fps_val   = 20.0 / (time.time() - fps_timer)
#                 fps_timer = time.time()
#                 fps_count = 0

#             # ── Detect faces (haar) ───────────────────────────
#             gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             rects = face_cascade.detectMultiScale(
#                 gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
#             )
#             rects_list = [tuple(r) for r in rects] if len(rects) else []

#             # ── Update centroid tracker ───────────────────────
#             id_map = tracker.update(rects_list)
#             smoother.cleanup(set(id_map.keys()))

#             # ── OpenCV eye cascade attention ──────────────────
#             ear_map: Dict[int, int] = {}   # face_id → eye count (0/1/2)

#             # ── Per-face processing ───────────────────────────
#             face_data: List[dict] = []

#             for rect_idx, (x, y, w_r, h_r) in enumerate(rects_list):
#                 cx, cy = x + w_r // 2, y + h_r // 2

#                 # Find best matching face ID
#                 if id_map:
#                     face_id = min(
#                         id_map,
#                         key=lambda k: float(np.linalg.norm(id_map[k] - np.array([cx, cy])))
#                     )
#                 else:
#                     face_id = 0

#                 # ── Predict & smooth ──────────────────────────
#                 face_gray   = gray[y: y+h_r, x: x+w_r]
#                 raw_pred    = model.predict(preprocess(face_gray), verbose=0)[0]
#                 smooth_pred = smoother.update(face_id, raw_pred)

#                 label_idx  = int(np.argmax(smooth_pred))
#                 emotion    = LABELS[label_idx]
#                 confidence = float(smooth_pred[label_idx])

#                 # ── Dominant emotion timer ────────────────────
#                 if dom_emo.get(face_id) != emotion:
#                     dom_emo[face_id]   = emotion
#                     dom_start[face_id] = time.time()
#                 hold_secs = time.time() - dom_start.get(face_id, time.time())

#                 # ── Attention (eye cascade) ──────────────────
#                 face_roi  = gray[y: y+h_r, x: x+w_r]
#                 eye_count = detect_eyes(eye_cascade, face_roi)
#                 ear_map[face_id] = eye_count
#                 attentive = eye_count >= 2

#                 # ── Audio feedback ────────────────────────────
#                 audio.notify(emotion)

#                 # ── CSV logging ───────────────────────────────
#                 logger.log(face_id, emotion, confidence)

#                 # ── Auto-snapshot ─────────────────────────────
#                 if confidence >= SNAP_CONFIDENCE and last_snap.get(face_id) != emotion:
#                     last_snap[face_id] = emotion
#                     fname = os.path.join(
#                         SNAPSHOTS_DIR,
#                         f"{emotion}_{datetime.now().strftime('%H%M%S')}_f{face_id}.jpg"
#                     )
#                     cv2.imwrite(fname, frame)
#                     print(f"[Snapshot] {fname}")

#                 # ── History ───────────────────────────────────
#                 history.append((emotion, confidence))

#                 # ── Color interpolation ───────────────────────
#                 target = EMOTION_COLORS[emotion]
#                 prev   = prev_col.get(face_id, target)
#                 color  = UIRenderer.lerp_color(prev, target, 0.20)
#                 prev_col[face_id] = color

#                 # ── Draw: pulse ring ──────────────────────────
#                 UIRenderer.pulse_ring(frame, cx, cy, max(w_r, h_r) // 2 + 8,
#                                       tick + face_id * 11, color)

#                 # ── Draw: rounded face bounding box ───────────
#                 UIRenderer.rounded_rect(frame, (x, y), (x+w_r, y+h_r),
#                                         color, r=14, thickness=2)

#                 # ── Draw: emoji (PIL) ─────────────────────────
#                 emoji = EMOTION_EMOJI.get(emotion, "")
#                 UIRenderer.draw_emoji(frame, emoji, x + w_r//2 - 17, max(y - 46, 2))

#                 # ── Draw: label badge ─────────────────────────
#                 badge_txt = f"#{face_id} {emotion}  {confidence*100:.0f}%"
#                 (tw, th), _ = cv2.getTextSize(badge_txt, CV_FONT, 0.46, 1)
#                 badge_y = max(y - 8, th + 8)
#                 UIRenderer.rounded_rect(frame,
#                     (x, badge_y - th - 6), (x + tw + 10, badge_y + 3),
#                     color, r=6, filled=True)
#                 cv2.putText(frame, badge_txt, (x+5, badge_y - 2),
#                             CV_FONT, 0.46, (10, 10, 10), 1, cv2.LINE_AA)

#                 # ── Draw: attention dot ───────────────────────
#                 dot_col = (0, 210, 80) if attentive else (0, 70, 220)
#                 cv2.circle(frame, (x + w_r - 10, y + 10), 7, dot_col, -1, cv2.LINE_AA)
#                 cv2.circle(frame, (x + w_r - 10, y + 10), 7, (255,255,255), 1, cv2.LINE_AA)

#                 # ── Draw: hold timer below box ────────────────
#                 UIRenderer.hold_timer(frame, x+2, y+h_r+16, emotion, hold_secs, color)

#                 # ── Draw: eye count (small, below box) ───────
#                 eye_count = ear_map.get(face_id, 0)
#                 cv2.putText(frame, f"Eyes:{eye_count}/2", (x+2, y+h_r+30),
#                             CV_FONT, 0.28, (130, 130, 160), 1, cv2.LINE_AA)

#                 face_data.append({
#                     "id": face_id, "emotion": emotion,
#                     "confidence": confidence, "timer": hold_secs,
#                     "attention": attentive,
#                 })

#                 # ── Live bar graph (first face, bottom-left) ──
#                 if show_bars and rect_idx == 0:
#                     UIRenderer.bar_graph(frame, smooth_pred,
#                                          ox=10, oy=frame_h - 168)

#             # ── Glassmorphism sidebar ─────────────────────────
#             if show_sidebar:
#                 UIRenderer.sidebar(frame, face_data, history, frame_h, frame_w)

#             # ── FPS + face count overlay ──────────────────────
#             info_txt = f"FPS:{fps_val:.1f}  Faces:{len(rects_list)}"
#             cv2.putText(frame, info_txt, (10, 22),
#                         CV_FONT, 0.46, (160, 220, 160), 1, cv2.LINE_AA)

#             cv2.imshow("Emotion Detector  [q/ESC quit | s snapshot | b bars | p panel]", frame)
#             key = cv2.waitKey(1) & 0xFF

#             if key in (ord("q"), 27):           # quit
#                 break
#             elif key == ord("s"):               # manual snapshot
#                 manual_path = os.path.join(SNAPSHOTS_DIR, f"manual_{datetime.now().strftime('%H%M%S%f')}.jpg")
#                 cv2.imwrite(manual_path, frame)
#                 print(f"[Manual Snapshot] {manual_path}")
#             elif key == ord("b"):               # toggle bar graph
#                 show_bars = not show_bars
#             elif key == ord("p"):               # toggle sidebar panel
#                 show_sidebar = not show_sidebar

#     finally:
#         webcam.release()
#         cv2.destroyAllWindows()
#         logger.close()
#         logger.save_pie_chart()
#         audio.cleanup()
#         print("\n[Done] CSV log and session pie chart saved. Goodbye!")


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # ENTRY POINT
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# if __name__ == "__main__":
#     try:
#         model = load_model("emotiondetector.json", "emotiondetector.h5")
        
#         # ⬇️ CHOOSE YOUR CAMERA SOURCE HERE ⬇️
        
#         # Option A: Standard Laptop Webcam
#         run(model, camera_source=0)
        
#         # Option B: DroidCam Stream
#         # Replace the IP and Port below with the exact ones shown in your DroidCam app!
#         # droidcam_url = "http://192.168.1.9:4747/video" 
#         # run(model, camera_source=droidcam_url)
        
#     except (FileNotFoundError, RuntimeError) as e:
#         print(f"[ERROR] {e}")
#         sys.exit(1)