"""
AI Virtual Painter — Extended Version (with Toolbar UI)
=====================================================
Features:
- Hand tracking (MediaPipe) with visible skeleton overlay
- Modes: Free draw, Rectangle, Circle
- Dynamic brush size via pinch (thumb–index distance)
- **Toolbar of square buttons with TEXT** (no emojis): colors, brush sizes, Undo, Clear, Save
- Gesture-based UI click: **hover a button with index tip + pinch (thumb/index)** to select
- Gesture commands still work: 👍 save, 🖐 clear, four-up undo
- HUD overlay (mode, color, brush size)

Install deps (Python 3.9+ recommended):
    pip install opencv-python mediapipe numpy pillow

Run:
    python main.py

Controls / Gestures:
- Index finger only: Free draw (paint with index tip)
- Pinch (thumb+index) while hovering a toolbar button: "click" that button
- Circle: While in index_mode, pinch to preview circle; release to commit
- Rectangle: Thumb+index+middle (rect_arm) pinch-drag to preview; release to commit
- Fist/four-up/palm mapped as before (undo/save/clear via gestures). Buttons also available.

Notes:
- Drawing is **disabled** when your index fingertip is inside the toolbar region (top band)
- Undo triggers once per hold (debounced)
- Shapes finalize when you leave the shape gesture or release pinch
- History stack replays canvas on undo for correctness
"""

from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np
import time, os, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image

# =========================
# Utilities
# =========================
def dist(p1: Tuple[int,int], p2: Tuple[int,int]) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def clamp(v, lo, hi): return max(lo, min(hi, v))

# =========================
# Drawing primitives & history
# =========================
@dataclass
class Stroke:
    pts: List[Tuple[int,int]]
    color: Tuple[int,int,int]
    size: int

@dataclass
class RectShape:
    p1: Tuple[int,int]
    p2: Tuple[int,int]
    color: Tuple[int,int,int]
    size: int

@dataclass
class CircleShape:
    center: Tuple[int,int]
    radius: int
    color: Tuple[int,int,int]
    size: int

Action = Dict[str, Any]  # {'type': 'stroke'|'rect'|'circle', 'data': ...}

class History:
    def __init__(self):
        self.actions: List[Action] = []

    def push(self, a: Action): self.actions.append(a)

    def pop(self) -> Optional[Action]:
        if self.actions: return self.actions.pop()
        return None

    def redraw(self, canvas: np.ndarray):
        canvas[:] = 0
        for a in self.actions:
            t = a['type']; d = a['data']
            if t == 'stroke':
                for i in range(1, len(d.pts)):
                    cv2.line(canvas, d.pts[i-1], d.pts[i], d.color, d.size, cv2.LINE_AA)
            elif t == 'rect':
                cv2.rectangle(canvas, d.p1, d.p2, d.color, d.size)
            elif t == 'circle':
                cv2.circle(canvas, d.center, d.radius, d.color, d.size)

# =========================
# MediaPipe helpers & gestures
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def finger_states(hand_landmarks, handed_label: str) -> Dict[str, bool]:
    lm = hand_landmarks.landmark
    # thumb uses x, others use y
    thumb_tip, thumb_ip, thumb_mcp = lm[4].x, lm[3].x, lm[2].x
    if handed_label == 'Right':
        thumb_up = thumb_tip > thumb_ip and thumb_tip > thumb_mcp
    else:
        thumb_up = thumb_tip < thumb_ip and thumb_tip < thumb_mcp

    def up(tip, pip): return lm[tip].y < lm[pip].y
    states = {
        'thumb': thumb_up,
        'index': up(8, 6),
        'middle': up(12, 10),
        'ring': up(16, 14),
        'pinky': up(20, 18)
    }
    return states

def classify_gesture(states: Dict[str,bool]) -> str:
    # exact sets for commands
    if states['thumb'] and not any(states[f] for f in ['index','middle','ring','pinky']):
        return 'thumbs_up'         # SAVE
    if all(states.values()):
        return 'palm'              # CLEAR
    if (not states['thumb']) and states['index'] and states['middle'] and states['ring'] and states['pinky']:
        return 'four_up'           # UNDO
    # drawing/shape modes
    if states['index'] and not any(states[f] for f in ['middle','ring','pinky']):
        # index up; thumb can be either (pinch check done separately)
        return 'index_mode'        # DRAW / possibly CIRCLE (with pinch)
    if states['thumb'] and states['index'] and states['middle'] and not states['ring'] and not states['pinky']:
        return 'rect_arm'          # RECTANGLE (needs 3-finger pinch)
    return 'unknown'

# =========================
# Palette UI (kept for compatibility, unused by new toolbar)
# =========================
PALETTE = [
    (0,0,0),      # black
    (255,255,255),# white
    (255,0,0),    # blue (BGR)
    (0,255,0),    # green
    (0,0,255),    # red
    (0,255,255),  # yellow
    (255,0,255),  # magenta
    (255,255,0),  # cyan
]

def draw_palette(frame: np.ndarray, selected_idx: int) -> List[Tuple[int,int,int,int]]:
    # Old palette (not displayed anymore). Return empty list to disable hover-pick conflicts.
    return []

def hit_palette(pt: Tuple[int,int], boxes) -> Optional[int]:
    return None

# =========================
# Toolbar (Squares with TEXT)
# =========================
@dataclass
class Button:
    id: str
    rect: Tuple[int,int,int,int]  # x,y,w,h
    label: str
    kind: str                     # 'color' | 'brush' | 'action'
    value: Any

# =========================
# Main App
# =========================
class PainterApp:
    def __init__(self, cam=0, w=1280, h=720):
        self.cap = cv2.VideoCapture(cam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.w, self.h = w, h

        # Allow up to 2 hands
        self.hands = mp_hands.Hands(model_complexity=1, max_num_hands=2,
                                    min_detection_confidence=0.6, min_tracking_confidence=0.5)

        self.canvas   = np.zeros((h,w,3), np.uint8)       # committed art
        self.preview  = np.zeros((h,w,3), np.uint8)       # ghost shapes only (cleared every frame)
        self.history  = History()

        # drawing state (global defaults)
        self.color_idx = 4  # default red (BGR) — index in PALETTE (kept for HUD swatch)
        self.color     = PALETTE[self.color_idx]
        self.brush     = 12
        self.eraser    = False

        # per-hand state: Left and Right
        self.hand_state = {
            'Left':  {'live_stroke': None, 'last_draw_pt': None, 'circle_active': False, 'circle_center': None, 'rect_active': False, 'rect_start': None},
            'Right': {'live_stroke': None, 'last_draw_pt': None, 'circle_active': False, 'circle_center': None, 'rect_active': False, 'rect_start': None}
        }

        # debouncing
        self.last_save = 0.0; self.save_cd = 1.2
        self.last_clear = 0.0; self.clear_cd = 1.0
        self.last_undo = 0.0; self.undo_cd = 0.6
        self.last_click = 0.0; self.click_cd = 0.45  # toolbar click debounce

        # toolbar
        self.toolbar_h = 78
        self.buttons: List[Button] = []
        self._setup_toolbar()

        os.makedirs('artworks', exist_ok=True)

    # ---------- Toolbar helpers ----------
    def _setup_toolbar(self):
        x = 8; y = 8; bw = 84; bh = self.toolbar_h - 16; gap = 8
        btns: List[Button] = []
        # 8 colors matching PALETTE order (labels are short TEXT)
        labels = ['BLK','WHT','BLU','GRN','RED','YLW','MAG','CYN']
        for i, col in enumerate(PALETTE):
            btns.append(Button(id=f"col_{i}", rect=(x, y, bw, bh), label=labels[i], kind='color', value=i))
            x += bw + gap
        # Brush sizes
        for size, lab in [(6,'S'), (12,'M'), (24,'L')]:
            btns.append(Button(id=f"brush_{size}", rect=(x, y, bw, bh), label=f"BR {lab}", kind='brush', value=size))
            x += bw + gap
        # Actions
        for act, lab in [('undo','UNDO'), ('clear','CLEAR'), ('save','SAVE')]:
            btns.append(Button(id=f"act_{act}", rect=(x, y, bw, bh), label=lab, kind='action', value=act))
            x += bw + gap
        self.buttons = btns

    def _hit_toolbar(self, pt: Optional[Tuple[int,int]]) -> Optional[Button]:
        if pt is None: return None
        x, y = pt
        for b in self.buttons:
            bx, by, bw, bh = b.rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return b
        return None

    def _render_toolbar(self, frame: np.ndarray, hover: Optional[Button]=None):
        # background band
        cv2.rectangle(frame, (0,0), (self.w, self.toolbar_h), (35,35,35), -1)
        cv2.line(frame, (0,self.toolbar_h), (self.w, self.toolbar_h), (80,80,80), 2)
        # draw each button
        for b in self.buttons:
            bx, by, bw, bh = b.rect
            # fill color button with its color; others with light gray
            if b.kind == 'color':
                fill = PALETTE[b.value]
            else:
                fill = (210,210,210)
            cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), fill, -1)
            # selection outline
            selected = (b.kind=='color' and b.value==self.color_idx) or (b.kind=='brush' and b.value==self.brush)
            outline = (0,255,0) if (hover is not None and hover.id==b.id) else ((255,255,255) if selected else (90,90,90))
            cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), outline, 2)
            # centered label (black text on light, white on dark colors)
            (tw, th), bl = cv2.getTextSize(b.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = bx + (bw - tw)//2
            ty = by + (bh + th)//2
            text_color = (0,0,0)
            if b.kind=='color':
                # choose white text for dark fills
                if sum(PALETTE[b.value]) < 255:
                    text_color = (255,255,255)
            cv2.putText(frame, b.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    def _click_toolbar(self, btn: Button, frame_for_save: np.ndarray):
        if btn.kind == 'color':
            self.color_idx = int(btn.value)
            self.color = PALETTE[self.color_idx]
            self.eraser = (self.color == (0,0,0))
        elif btn.kind == 'brush':
            self.brush = int(btn.value)
        elif btn.kind == 'action':
            if btn.value == 'undo':
                # commit any live strokes first (for both hands)
                self._commit_live_stroke(None)
                self._undo()
                # cancel shapes for both
                for h in self.hand_state.values():
                    h['circle_active']=False; h['rect_active']=False; h['circle_center']=None; h['rect_start']=None
            elif btn.value == 'clear':
                self._commit_live_stroke(None)
                self.history.actions.clear()
                self.canvas[:] = 0
                # cancel shapes
                for h in self.hand_state.values():
                    h['circle_active']=False; h['rect_active']=False; h['circle_center']=None; h['rect_start']=None
                print("🧹 Cleared canvas (button)")
            elif btn.value == 'save':
                self._commit_live_stroke(None); self._save(frame_for_save)

    # ---------- Painter internals ----------
    def _blend(self, frame):
        # frame + canvas + preview
        out = cv2.addWeighted(frame, 1.0, self.canvas, 1.0, 0)
        out = cv2.addWeighted(out,   1.0, self.preview, 0.9, 0)
        return out

    def _hud(self, frame):
        # Mode text: show any active hand modes (simple summary)
        modes = []
        for hand_label, s in self.hand_state.items():
            if s['circle_active']:
                modes.append(f"{hand_label}:Circle")
            elif s['rect_active']:
                modes.append(f"{hand_label}:Rect")
            else:
                # check if drawing
                if s['live_stroke'] is not None:
                    modes.append(f"{hand_label}:Draw")
        mode = ', '.join(modes) if modes else 'Draw'
        cv2.rectangle(frame, (10,self.h-78), (620,self.h-12), (0,0,0), -1)
        cv2.putText(frame, f"Mode: {mode}", (20,self.h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Brush: {self.brush}", (20,self.h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        # color swatch
        cv2.rectangle(frame,(630,self.h-78),(700,self.h-12), self.color, -1)
        cv2.rectangle(frame,(630,self.h-78),(700,self.h-12), (255,255,255), 2)

    def _commit_live_stroke(self, handed: Optional[str]):
        """If `handed` is None commit all hands' live strokes; otherwise commit only that hand."""
        if handed is None:
            for label in self.hand_state.keys():
                hand_data = self.hand_state[label]
                if hand_data['live_stroke'] and len(hand_data['live_stroke'].pts) > 1:
                    self.history.push({'type':'stroke','data': hand_data['live_stroke']})
                hand_data['live_stroke']=None
                hand_data['last_draw_pt']=None
        else:
            if handed not in self.hand_state: return
            hand_data = self.hand_state[handed]
            if hand_data['live_stroke'] and len(hand_data['live_stroke'].pts) > 1:
                self.history.push({'type':'stroke','data': hand_data['live_stroke']})
            hand_data['live_stroke']=None
            hand_data['last_draw_pt']=None

    def _undo(self):
        popped = self.history.pop()
        if popped is not None:
            self.history.redraw(self.canvas)

    def _save(self, frame_bgr):
        ts = int(time.time())
        out_path = f"artworks/art_{ts}.png"
        blended = self._blend(frame_bgr.copy())
        Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)).save(out_path)
        print(f"✅ Saved: {out_path}")

    def run(self):
        prev = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            self.preview[:] = 0  # clear ghost layer

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            # --- Hand skeleton overlay (drawn on frame so it's visible) ---
            hover_btn: Optional[Button] = None
            any_ui_mode = False  # if any hand is in toolbar area

            if res.multi_hand_landmarks and res.multi_handedness:
                # Process all hands (max 2)
                for hand_idx, (hand_lms, handedness) in enumerate(zip(res.multi_hand_landmarks, res.multi_handedness)):
                    handed = handedness.classification[0].label  # "Left" or "Right"
                    lm = hand_lms.landmark
                    # pixel points
                    points = [(int(lm[i].x*self.w), int(lm[i].y*self.h)) for i in range(21)]
                    idx_pt, mid_pt, th_pt = points[8], points[12], points[4]

                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # Compute bounding box (padded)
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x_min, x_max = max(0, min(xs)-20), min(self.w, max(xs)+20)
                    y_min, y_max = max(0, min(ys)-20), min(self.h, max(ys)+20)

                    # Draw rectangle and label
                    box_color = (0,200,255) if handed == 'Right' else (0,255,0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    cv2.putText(frame, handed, (x_min, max(12, y_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                    # Gesture & per-hand state
                    states = finger_states(hand_lms, handed)
                    gesture = classify_gesture(states)

                    # Toolbar hover & click: any hand can control toolbar
                    now = time.time()
                    pinch = (idx_pt is not None and th_pt is not None and dist(idx_pt, th_pt) < 40)
                    # Detect hover button under this hand
                    btn_under = self._hit_toolbar(idx_pt)
                    # prefer hover if first hit or if it's the same as current hover
                    if btn_under is not None:
                        hover_btn = btn_under
                    if idx_pt is not None and idx_pt[1] < self.toolbar_h:
                        any_ui_mode = True
                        # commit this hand's stroke when interacting with UI
                        self._commit_live_stroke(handed)
                        # cancel this hand's shape previews while hovering
                        self.hand_state[handed]['circle_active']=False
                        self.hand_state[handed]['rect_active']=False
                        self.hand_state[handed]['circle_center']=None
                        self.hand_state[handed]['rect_start']=None
                        if btn_under is not None and pinch and (now - self.last_click) > self.click_cd:
                            # Click triggered by this hand
                            self._click_toolbar(btn_under, frame)
                            self.last_click = now

                    # --- Commands (debounced) --- (still available, per-hand)
                    if not any_ui_mode:
                        if gesture == 'thumbs_up' and (now - self.last_save) > self.save_cd:
                            self._commit_live_stroke(handed)
                            self._save(frame)
                            self.last_save = now

                        elif gesture == 'palm' and (now - self.last_clear) > self.clear_cd:
                            self._commit_live_stroke(handed)
                            self.history.actions.clear()
                            self.canvas[:] = 0
                            print("🧹 Cleared canvas")
                            self.last_clear = now
                            # cancel shapes for this hand
                            self.hand_state[handed]['circle_active']=False
                            self.hand_state[handed]['rect_active']=False
                            self.hand_state[handed]['circle_center']=None
                            self.hand_state[handed]['rect_start']=None

                        elif gesture == 'four_up' and (now - self.last_undo) > self.undo_cd:
                            self._commit_live_stroke(handed)
                            self._undo()
                            print("↩ Undo")
                            self.last_undo = now
                            # cancel shapes for this hand
                            self.hand_state[handed]['circle_active']=False
                            self.hand_state[handed]['rect_active']=False
                            self.hand_state[handed]['circle_center']=None
                            self.hand_state[handed]['rect_start']=None

                    # --- SHAPES & DRAWING (per-hand) ---
                    hand_data = self.hand_state.get(handed, None)
                    if hand_data is None:
                        # initialize if unexpected label
                        self.hand_state[handed] = {'live_stroke': None, 'last_draw_pt': None, 'circle_active': False, 'circle_center': None, 'rect_active': False, 'rect_start': None}
                        hand_data = self.hand_state[handed]

                    # Circle: pinch index+thumb -> hold to preview, release to commit
                    if not any_ui_mode:
                        if gesture == 'index_mode':
                            if idx_pt and th_pt and dist(idx_pt, th_pt) < 45:
                                # start or continue circle preview
                                if not hand_data['circle_active']:
                                    hand_data['circle_active'] = True
                                    hand_data['circle_center'] = idx_pt
                                r = max(1, int(dist(hand_data['circle_center'], idx_pt)))
                                cv2.circle(self.preview, hand_data['circle_center'], r, self.color, max(2, self.brush))
                                # while pinching, do NOT free-draw
                                hand_data['live_stroke'] = None
                                hand_data['last_draw_pt'] = None
                            else:
                                # if circle was active and pinch released -> commit
                                if hand_data['circle_active'] and hand_data['circle_center'] and idx_pt:
                                    r = max(1, int(dist(hand_data['circle_center'], idx_pt)))
                                    circ = CircleShape(hand_data['circle_center'], r, self.color, max(2, self.brush))
                                    self.history.push({'type':'circle','data':circ})
                                    cv2.circle(self.canvas, circ.center, circ.radius, circ.color, circ.size)
                                hand_data['circle_active']=False
                                hand_data['circle_center']=None
                                # If not pinching, proceed with free drawing (index only)
                                if states['index'] and not any(states[f] for f in ['middle','ring','pinky']):
                                    if hand_data['last_draw_pt'] is None:
                                        hand_data['last_draw_pt'] = idx_pt
                                        hand_data['live_stroke'] = Stroke([hand_data['last_draw_pt']], self.color, self.brush)
                                    if idx_pt:
                                        cv2.line(self.canvas, hand_data['last_draw_pt'], idx_pt, self.color, self.brush, cv2.LINE_AA)
                                        hand_data['live_stroke'].pts.append(idx_pt)
                                        hand_data['last_draw_pt'] = idx_pt
                        else:
                            # leaving index_mode: finalize any live stroke for that hand
                            self._commit_live_stroke(handed)
                            # also if leaving while circle active without release, cancel preview
                            if hand_data['circle_active']:
                                hand_data['circle_active']=False; hand_data['circle_center']=None

                        # Rectangle: 3-finger (thumb+index+middle) pinch
                        if gesture == 'rect_arm':
                            if idx_pt and th_pt and dist(idx_pt, th_pt) < 48 and mid_pt:
                                if not hand_data['rect_active']:
                                    hand_data['rect_active'] = True
                                    hand_data['rect_start'] = idx_pt  # start at index position on pinch start
                                # preview rectangle to current index tip
                                cv2.rectangle(self.preview, hand_data['rect_start'], idx_pt, self.color, max(2, self.brush))
                                hand_data['live_stroke'] = None
                                hand_data['last_draw_pt'] = None
                            else:
                                # pinch released while was active -> commit
                                if hand_data['rect_active'] and hand_data['rect_start'] and idx_pt:
                                    rect = RectShape(hand_data['rect_start'], idx_pt, self.color, max(2, self.brush))
                                    self.history.push({'type':'rect','data':rect})
                                    cv2.rectangle(self.canvas, rect.p1, rect.p2, rect.color, rect.size)
                                hand_data['rect_active']=False
                                hand_data['rect_start']=None
                        else:
                            if hand_data['rect_active']:
                                hand_data['rect_active']=False; hand_data['rect_start']=None

            else:
                # no hand: finalize any live stroke and cancel previews for all hands
                self._commit_live_stroke(None)
                for h in self.hand_state.values():
                    h['circle_active']=False; h['rect_active']=False
                    h['circle_center']=None; h['rect_start']=None
                hover_btn = None
                any_ui_mode = False

            # Compose & HUD
            # Render toolbar (after any changes so highlighting is up-to-date)
            self._render_toolbar(frame, hover_btn)

            out = self._blend(frame)
            self._hud(out)

            # FPS (optional)
            now = time.time()
            fps = 1.0 / (now - prev) if now>prev else 0.0
            prev = now
            cv2.putText(out, f"FPS: {fps:.1f}", (self.w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(out, f"FPS: {fps:.1f}", (self.w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

            cv2.imshow("AI Virtual Painter — Gestures + Preview (Multi-Hand)", out)
            if (cv2.waitKey(1) & 0xFF) == 27: break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    PainterApp().run()