"""
The real Scritchy Scratchy on macOS: see its window, move the mouse, read the
money, and snapshot or restore saves.

RealGame has the same interface as sim.ScratchSim, so a fly (or any policy)
can play either one:

    env.reset()              -> first frame
    env.frame()              -> H x W x 3 uint8 RGB of the game window
    env.act(x, y, button)    -> move the mouse; x, y in 0-1 window coordinates,
                                button 'up', 'down' (drag = scratch) or 'click'
    env.money()              -> money on screen, or None if unreadable

Safety: the fly never touches the mouse while another app is in front, and
moving the mouse yourself (or into a screen corner) raises FlyStopped.

Needs pyobjc-framework-Quartz and pyobjc-framework-Vision. The app you run it
from needs Screen Recording and Accessibility permission (System Settings >
Privacy & Security).

Usage:
    python code/scratch/game_io.py --selftest          # window, capture, OCR, cursor square
    python code/scratch/game_io.py --find-save         # where does the game keep its save?
    python code/scratch/game_io.py --snapshot NAME     # copy the current save aside
    python code/scratch/game_io.py --restore NAME      # put a snapshot back (relaunches the game)
"""

import argparse
import ctypes
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fly_screen import check_screen_permission, frontmost_app  # noqa: E402

GAME_OWNER = 'Scritchy Scratchy'          # process name that owns the window
BUNDLE_ID = 'com.DefaultCompany.2D-URP'   # yes, really (from the app's Info.plist)
STEAM_APP_ID = 3948120
SAVE_DIR = Path.home() / 'Library/Application Support/Lunch Money Games/Scritchy Scratchy'
PREFS_DOMAIN = 'unity.Lunch Money Games.Scritchy Scratchy'
PREFS_PLIST = Path.home() / 'Library/Preferences' / f'{PREFS_DOMAIN}.plist'

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'scratch'
SNAP_DIR = DATA_DIR / 'saves'
LAYOUT_JSON = DATA_DIR / 'layout.json'    # money rect etc., written after recon

USER_MOVE_PT = 30.0                       # mouse this far from where the fly left it: user took over
CORNER_PT = 4.0                           # mouse this close to a screen corner: stop
MONEY_RE = re.compile(r'\$\s*([\d,]*\.?\d+)\s*([KMBTkmbt]|Qa|Qi|Sx|Sp|Oc|No|Dc)?\b')
NUMBER_RE = re.compile(r'([\d,]*\.?\d+)\s*([KMBTkmbt]|Qa|Qi|Sx|Sp|Oc|No|Dc)?\b')
SUFFIX = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12, 'qa': 1e15, 'qi': 1e18,
          'sx': 1e21, 'sp': 1e24, 'oc': 1e27, 'no': 1e30, 'dc': 1e33}


class FlyStopped(Exception):
    """The user took the mouse back, or the game is not in front."""


# ============================================================================
# Permissions and the game window
# ============================================================================

def check_accessibility():
    """Posting mouse events needs Accessibility permission; warn if missing."""
    try:
        hi = ctypes.cdll.LoadLibrary(
            '/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices')
        hi.AXIsProcessTrusted.restype = ctypes.c_bool
        if not hi.AXIsProcessTrusted():
            print('Accessibility permission is missing, so the fly cannot move the mouse.\n'
                  'Allow it for this terminal in System Settings > Privacy & Security > '
                  'Accessibility, then restart it.')
            return False
    except (OSError, AttributeError):
        pass
    return True


def find_window():
    """The game's main window as dict(id, pid, x, y, w, h) in screen points, or None."""
    import Quartz
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    best = None
    for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
        if w.get('kCGWindowOwnerName') != GAME_OWNER or w.get('kCGWindowLayer', 1) != 0:
            continue
        b = w['kCGWindowBounds']
        win = dict(id=int(w['kCGWindowNumber']), pid=int(w['kCGWindowOwnerPID']),
                   x=float(b['X']), y=float(b['Y']), w=float(b['Width']), h=float(b['Height']))
        if best is None or win['w'] * win['h'] > best['w'] * best['h']:
            best = win
    return best


def game_in_front():
    name, _, _ = frontmost_app()
    return name is not None and GAME_OWNER.replace(' ', '').lower() in name.replace(' ', '').lower()


# ============================================================================
# Mouse
# ============================================================================

class Mouse:
    """Posts mouse events inside the game window, and notices when you take over."""

    def __init__(self, win, exclude=()):
        self.win = win
        self.exclude = list(exclude)   # (x0, y0, x1, y1) in 0-1 window coordinates
        self.down = False
        self.last = None               # last position we put the mouse at, screen points

    def to_screen(self, x, y):
        w = self.win
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        return w['x'] + x * (w['w'] - 1), w['y'] + y * (w['h'] - 1)

    def excluded(self, x, y):
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self.exclude)

    def where(self):
        import Quartz
        p = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return p.x, p.y

    def check_user(self):
        """Raise FlyStopped if the user moved the mouse or the game lost focus."""
        import Quartz
        x, y = self.where()
        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        W, H = bounds.size.width, bounds.size.height
        if min(x, W - 1 - x) < CORNER_PT and min(y, H - 1 - y) < CORNER_PT:
            self.release()
            raise FlyStopped('mouse in a screen corner')
        if self.last is not None and math.hypot(x - self.last[0], y - self.last[1]) > USER_MOVE_PT:
            self.release()
            raise FlyStopped('you moved the mouse')
        if not game_in_front():
            self.release()
            raise FlyStopped('the game is not the frontmost app')

    def _post(self, kind, sx, sy):
        import Quartz
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (sx, sy), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        self.last = (sx, sy)

    def move(self, x, y, down):
        """Move to window point (x, y) with the button held (scratch) or not."""
        import Quartz
        self.check_user()
        if self.excluded(x, y):
            down = False
        sx, sy = self.to_screen(x, y)
        if down and not self.down:
            self._post(Quartz.kCGEventLeftMouseDown, sx, sy)
            self.down = True
        elif not down and self.down:
            self._post(Quartz.kCGEventLeftMouseUp, sx, sy)
            self.down = False
        self._post(Quartz.kCGEventLeftMouseDragged if self.down else Quartz.kCGEventMouseMoved, sx, sy)

    def click(self, x, y):
        import Quartz
        self.check_user()
        if self.excluded(x, y):
            return False
        self.release()
        sx, sy = self.to_screen(x, y)
        self._post(Quartz.kCGEventMouseMoved, sx, sy)
        self._post(Quartz.kCGEventLeftMouseDown, sx, sy)
        time.sleep(0.03)
        self._post(Quartz.kCGEventLeftMouseUp, sx, sy)
        return True

    def release(self):
        if self.down:
            import Quartz
            sx, sy = self.last if self.last else self.where()
            self._post(Quartz.kCGEventLeftMouseUp, sx, sy)
            self.down = False


# ============================================================================
# Text on screen (Apple Vision OCR)
# ============================================================================

def read_text(rgb, fast=False):
    """OCR an RGB image. Returns [(text, confidence, (x0, y0, x1, y1))] with
    the box in 0-1 image coordinates, origin top left."""
    import Quartz
    import Vision
    h, w = rgb.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    data = rgba.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    cg = Quartz.CGImageCreate(w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
                              Quartz.kCGImageAlphaNoneSkipLast, provider, None, False,
                              Quartz.kCGRenderingIntentDefault)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast
                             else Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        return []
    out = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        (bx, by), (bw, bh) = obs.boundingBox()
        out.append((str(cand[0].string()), float(cand[0].confidence()),
                    (bx, 1 - by - bh, bx + bw, 1 - by)))
    return out


def parse_money(text, need_dollar=True):
    """'$1,234.5K' -> 1234500.0; None if there is no amount.

    The game's money counter shows a coin icon, not '$', so OCR of just that
    counter uses need_dollar=False.
    """
    m = (MONEY_RE if need_dollar else NUMBER_RE).search(text)
    if not m:
        return None
    value = float(m.group(1).replace(',', ''))
    return value * SUFFIX.get((m.group(2) or '').lower(), 1.0)


# ============================================================================
# Saves
# ============================================================================

def save_files():
    """Every file the game might keep progress in, with modification times."""
    files = [p for p in SAVE_DIR.rglob('*') if p.is_file()] if SAVE_DIR.exists() else []
    if PREFS_PLIST.exists():
        files.append(PREFS_PLIST)
    return sorted(files, key=lambda p: -p.stat().st_mtime)


def saved_state():
    """The game's own save (save.json) as a dict, or None. It is only as fresh
    as the game's last autosave or quit."""
    try:
        return json.loads((SAVE_DIR / 'save.json').read_text())
    except (OSError, ValueError):
        return None


def quit_game(timeout=20.0):
    """Ask the game to quit (so it saves), and wait for its window to go."""
    subprocess.run(['osascript', '-e', f'tell application id "{BUNDLE_ID}" to quit'],
                   capture_output=True, timeout=10)
    t0 = time.time()
    while find_window() is not None and time.time() - t0 < timeout:
        time.sleep(0.5)
    time.sleep(1.0)
    return find_window() is None


def launch_game(timeout=90.0):
    """Start the game through Steam and wait for its window."""
    subprocess.run(['open', f'steam://rungameid/{STEAM_APP_ID}'], capture_output=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        win = find_window()
        if win and win['w'] > 200:
            time.sleep(3.0)                     # let it finish loading
            return find_window()
        time.sleep(1.0)
    raise RuntimeError('The game window did not appear. Is Steam running?')


def snapshot(name):
    """Copy the save folder and the game's preferences into data/scratch/saves/name."""
    dest = SNAP_DIR / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if SAVE_DIR.exists():
        shutil.copytree(SAVE_DIR, dest / 'save_dir')
    subprocess.run(['defaults', 'export', PREFS_DOMAIN, str(dest / 'prefs.plist')],
                   capture_output=True)
    return dest


def restore(name, relaunch=True):
    """Quit the game, put snapshot name back, and relaunch.

    The current save is first backed up as a timestamped snapshot, so a
    restore never loses progress.
    """
    src = SNAP_DIR / name
    if not src.exists():
        raise FileNotFoundError(f'No snapshot {src}')
    if find_window() is not None and not quit_game():
        raise RuntimeError('The game did not quit')
    snapshot(time.strftime('backup-%Y%m%d-%H%M%S'))
    if (src / 'save_dir').exists():
        if SAVE_DIR.exists():
            shutil.rmtree(SAVE_DIR)
        shutil.copytree(src / 'save_dir', SAVE_DIR)
    # defaults (not a file copy) so cfprefsd doesn't keep serving a cached copy
    subprocess.run(['defaults', 'delete', PREFS_DOMAIN], capture_output=True)
    if (src / 'prefs.plist').exists():
        subprocess.run(['defaults', 'import', PREFS_DOMAIN, str(src / 'prefs.plist')],
                       capture_output=True)
    return launch_game() if relaunch else None


# ============================================================================
# The environment
# ============================================================================

def load_layout():
    try:
        return json.loads(LAYOUT_JSON.read_text())
    except (OSError, ValueError):
        return {}


class RealGame:
    """The real game as an environment, same interface as sim.ScratchSim."""

    def __init__(self, monitor=1):
        import mss
        self.sct = (getattr(mss, 'MSS', None) or mss.mss)()
        self.layout = load_layout()
        self.win = None
        self.mouse = None
        self._money = None
        self._money_t = -math.inf

    def _attach(self):
        self.win = find_window()
        if self.win is None:
            raise RuntimeError(f'No "{GAME_OWNER}" window. Start the game first.')
        self.mouse = Mouse(self.win, [tuple(r) for r in self.layout.get('exclude', [])])

    def reset(self, save=None):
        if save:
            restore(save)
        self._attach()
        if not game_in_front():
            subprocess.run(['osascript', '-e', f'tell application id "{BUNDLE_ID}" to activate'],
                           capture_output=True)
            time.sleep(1.0)
        self.mouse.last = None
        return self.frame()

    def frame(self):
        w = self.win
        shot = self.sct.grab({'left': int(w['x']), 'top': int(w['y']),
                              'width': int(w['w']), 'height': int(w['h'])})
        img = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return np.ascontiguousarray(img[..., 2::-1])

    def act(self, x, y, button='up'):
        if button == 'click':
            return self.mouse.click(x, y)
        self.mouse.move(x, y, button == 'down')
        return True

    def money(self, max_age=1.0):
        """Money on the counter (OCR at most every max_age s), or None.

        Vision misreads the counter's pixel font when cropped tightly, so the
        layout's money_search area is read and the number whose box centre
        falls in the money rect is kept.
        """
        now = time.time()
        if now - self._money_t < max_age:
            return self._money
        img = self.frame()
        h, w = img.shape[:2]
        sx0, sy0, sx1, sy1 = self.layout.get('money_search', (0.0, 0.0, 1.0, 1.0))
        mx0, my0, mx1, my1 = self.layout.get('money', (0.0, 0.0, 1.0, 1.0))
        crop = img[int(sy0 * h):int(sy1 * h), int(sx0 * w):int(sx1 * w)]
        self._money = None
        for text, _, (x0, y0, x1, y1) in read_text(crop):
            cx = sx0 + (x0 + x1) / 2 * (sx1 - sx0)
            cy = sy0 + (y0 + y1) / 2 * (sy1 - sy0)
            if mx0 <= cx <= mx1 and my0 <= cy <= my1:
                value = parse_money(text, need_dollar=False)
                if value is not None:
                    self._money = value
                    break
        self._money_t = now
        return self._money

    def close(self):
        if self.mouse:
            self.mouse.release()


# ============================================================================
# Command line
# ============================================================================

def selftest():
    from PIL import Image
    check_screen_permission()
    check_accessibility()
    win = find_window()
    if win is None:
        sys.exit(f'No "{GAME_OWNER}" window found. Start the game and try again.')
    print(f'Window: {win}')
    env = RealGame()
    frame = env.reset()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / 'selftest.png'
    Image.fromarray(frame).save(path)
    print(f'Captured {frame.shape[1]}x{frame.shape[0]} px -> {path}  (mean brightness {frame.mean():.0f})')
    t0 = time.time()
    texts = read_text(frame)
    print(f'OCR found {len(texts)} text boxes in {time.time() - t0:.2f}s:')
    for text, conf, box in texts:
        money = parse_money(text)
        print(f'  {text!r:40s} conf {conf:.2f}  box {tuple(round(b, 3) for b in box)}'
              + (f'  -> ${money:,.2f}' if money is not None else ''))
    print('Moving the cursor in a square (no clicks). Move the mouse yourself to stop it.')
    try:
        for k in range(81):
            a = 2 * math.pi * k / 80
            env.act(0.5 + 0.2 * math.cos(a), 0.5 + 0.2 * math.sin(a), 'up')
            time.sleep(0.02)
        print('Cursor test done.')
    except FlyStopped as e:
        print(f'Stopped: {e}')
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Real Scritchy Scratchy: window, mouse, OCR, saves')
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--find-save', action='store_true')
    parser.add_argument('--snapshot', metavar='NAME')
    parser.add_argument('--restore', metavar='NAME')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.find_save:
        files = save_files()
        if not files:
            print('No save files yet. Play for a minute, quit the game, and try again.')
        for p in files:
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))}  '
                  f'{p.stat().st_size:9d} B  {p}')
    elif args.snapshot:
        print(f'Snapshot saved to {snapshot(args.snapshot)}. Quit the game first so it has saved.')
    elif args.restore:
        print(f'Restored {args.restore}; game window {restore(args.restore)}')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
