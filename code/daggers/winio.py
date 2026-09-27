"""
Windows I/O for Fly Daggers: find and watch the Devil Daggers window, hear
the player's keyboard and mouse, and (in play.py) press them for the fly.

- The player's input comes from raw input (WM_INPUT) on a hidden window, so
  mouse motion is the game's own raw counts even while the cursor is locked
  and hidden, and it arrives whichever window has focus. Events the fly
  injects with SendInput arrive with no device handle and are told apart.
- The fly's input goes through SendInput: scan codes for keys, relative
  motion for the mouse, so the game reads it as it reads a person's.
- The game is "in play" when its window has focus and the cursor is hidden;
  menus and the death screen show the cursor. Recording and the fly's
  controls both pause outside play.

Hotkeys (any window): F9 pauses or resumes, F10 stops.
"""

import ctypes
import queue
import threading
import time
from ctypes import wintypes

import numpy as np

GAME_TITLE = 'devil daggers'
VK_F9, VK_F10 = 0x78, 0x79
VK_R = 0x52                      # Devil Daggers: restart the run
INJECTED_TAG = 0x46_4C_59_44     # 'FLYD' in dwExtraInfo of the fly's own SendInput events

user32 = ctypes.WinDLL('user32', use_last_error=True)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindow.argtypes = [wintypes.HWND]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID,
                                   ctypes.POINTER(wintypes.UINT), wintypes.UINT]
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE


def make_dpi_aware():
    """Work in physical pixels, as screen capture does."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()


# ============================================================================
# The game window
# ============================================================================

def _title(hwnd):
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def find_game_window(title=GAME_TITLE):
    """Handle of the visible top-level window titled Devil Daggers, or None."""
    found = []

    def check(hwnd, _):
        if user32.IsWindowVisible(hwnd) and _title(hwnd).strip().lower() == title:
            found.append(hwnd)
        return True

    user32.EnumWindows(WNDENUMPROC(check), 0)
    return found[0] if found else None


def client_rect(hwnd):
    """The window's drawing area in screen pixels, as an mss monitor dict."""
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return {'left': pt.x, 'top': pt.y, 'width': r.right - r.left, 'height': r.bottom - r.top}


class CURSORINFO(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('flags', wintypes.DWORD),
                ('hCursor', wintypes.HANDLE), ('ptScreenPos', wintypes.POINT)]


def cursor_visible():
    ci = CURSORINFO(cbSize=ctypes.sizeof(CURSORINFO))
    if not user32.GetCursorInfo(ctypes.byref(ci)):
        return True
    return bool(ci.flags & 1) and bool(ci.hCursor)


class GameState:
    """Is Devil Daggers running, focused, and in play (cursor hidden)?"""

    def __init__(self, check_cursor=True):
        self.check_cursor = check_cursor
        self.hwnd = None
        self._next_search = 0.0

    def poll(self):
        """Returns (hwnd or None, focused, in_play)."""
        now = time.perf_counter()
        if self.hwnd is not None and not user32.IsWindow(self.hwnd):
            self.hwnd = None
        if self.hwnd is None and now >= self._next_search:
            self.hwnd = find_game_window()
            self._next_search = now + 1.0
        if self.hwnd is None:
            return None, False, False
        focused = user32.GetForegroundWindow() == self.hwnd
        in_play = focused and (not self.check_cursor or not cursor_visible())
        return self.hwnd, focused, in_play


gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
                                   ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, ctypes.c_int]
gdi32.SetBrushOrgEx.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
gdi32.StretchBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             wintypes.DWORD]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
SRCCOPY, HALFTONE = 0x00CC0020, 4


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD),
                ('biCompression', wintypes.DWORD), ('biSizeImage', wintypes.DWORD),
                ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD)]


class GameCamera:
    """Grabs the game's drawing area shrunk to size (w, h), RGB uint8.

    GDI's StretchBlt in HALFTONE mode (area averaging) copies the screen
    straight into a small bitmap at twice the size, which is then averaged
    2x2, so the full-resolution frame never reaches Python.
    """

    OVERSAMPLE = 2

    def __init__(self, size):
        self.w, self.h = size
        cw, ch = self.w * self.OVERSAMPLE, self.h * self.OVERSAMPLE
        self.screen_dc = user32.GetDC(None)
        self.mem_dc = gdi32.CreateCompatibleDC(self.screen_dc)
        header = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=cw,
                                  biHeight=-ch, biPlanes=1, biBitCount=32)   # top-down BGRA
        bits = ctypes.c_void_p()
        self.bitmap = gdi32.CreateDIBSection(self.screen_dc, ctypes.byref(header), 0,
                                             ctypes.byref(bits), None, 0)
        if not self.bitmap:
            raise OSError(f'CreateDIBSection error {ctypes.get_last_error()}')
        gdi32.SelectObject(self.mem_dc, self.bitmap)
        gdi32.SetStretchBltMode(self.mem_dc, HALFTONE)
        gdi32.SetBrushOrgEx(self.mem_dc, 0, 0, None)
        self.pixels = np.ctypeslib.as_array(
            (ctypes.c_uint8 * (cw * ch * 4)).from_address(bits.value)).reshape(ch, cw, 4)

    def grab(self, hwnd):
        rect = client_rect(hwnd)
        if rect['width'] < 16 or rect['height'] < 16:
            return None, rect
        s = self.OVERSAMPLE
        gdi32.StretchBlt(self.mem_dc, 0, 0, self.w * s, self.h * s, self.screen_dc,
                         rect['left'], rect['top'], rect['width'], rect['height'], SRCCOPY)
        gdi32.GdiFlush()
        bgr = self.pixels[..., :3].reshape(self.h, s, self.w, s, 3).mean(axis=(1, 3))
        return bgr[..., ::-1].round().astype(np.uint8), rect

    def close(self):
        gdi32.DeleteObject(self.bitmap)
        gdi32.DeleteDC(self.mem_dc)
        user32.ReleaseDC(None, self.screen_dc)


class FrameGrabber:
    """Watches the game on its own thread, fps times a second, so capture
    overlaps whatever the main thread is doing (the brain, in play.py).

    latest() waits for a look newer than the last one taken and returns
    (frame or None, capture time, hwnd, focused, in_play). A slow consumer
    skips frames rather than falling behind.
    """

    def __init__(self, size, fps, check_cursor=True):
        self.size, self.period = size, 1.0 / fps
        self.state = GameState(check_cursor)
        self._cond = threading.Condition()
        self._item = None
        self._seq = self._taken = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name='frame-grabber', daemon=True)
        self._thread.start()

    def _loop(self):
        camera = GameCamera(self.size)            # GDI objects belong to this thread
        next_t = time.perf_counter()
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                if next_t > now:
                    time.sleep(next_t - now)
                next_t = max(next_t + self.period, time.perf_counter() - self.period)
                hwnd, focused, in_play = self.state.poll()
                t = time.perf_counter()
                frame = camera.grab(hwnd)[0] if in_play else None
                with self._cond:
                    self._item = (frame, t, hwnd, focused, in_play)
                    self._seq += 1
                    self._cond.notify_all()
        finally:
            camera.close()

    def latest(self, timeout=1.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._taken, timeout):
                return None
            self._taken = self._seq
            return self._item

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# ============================================================================
# The player's input: raw input on a hidden window
# ============================================================================

WM_INPUT, WM_QUIT = 0x00FF, 0x0012
RID_INPUT = 0x10000003
RIM_TYPEMOUSE, RIM_TYPEKEYBOARD = 0, 1
RIDEV_INPUTSINK = 0x00000100
RI_KEY_BREAK = 0x01
MOUSE_MOVE_ABSOLUTE = 0x01
RI_MOUSE = {0x0001: ('left', True), 0x0002: ('left', False),
            0x0004: ('right', True), 0x0008: ('right', False)}


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [('usUsagePage', wintypes.USHORT), ('usUsage', wintypes.USHORT),
                ('dwFlags', wintypes.DWORD), ('hwndTarget', wintypes.HWND)]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [('dwType', wintypes.DWORD), ('dwSize', wintypes.DWORD),
                ('hDevice', wintypes.HANDLE), ('wParam', wintypes.WPARAM)]


class RAWMOUSE(ctypes.Structure):
    _fields_ = [('usFlags', wintypes.USHORT), ('_pad', wintypes.USHORT),
                ('usButtonFlags', wintypes.USHORT), ('usButtonData', wintypes.USHORT),
                ('ulRawButtons', wintypes.ULONG), ('lLastX', wintypes.LONG),
                ('lLastY', wintypes.LONG), ('ulExtraInformation', wintypes.ULONG)]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [('MakeCode', wintypes.USHORT), ('Flags', wintypes.USHORT),
                ('Reserved', wintypes.USHORT), ('VKey', wintypes.USHORT),
                ('Message', wintypes.UINT), ('ExtraInformation', wintypes.ULONG)]


class _RAWDATA(ctypes.Union):
    _fields_ = [('mouse', RAWMOUSE), ('keyboard', RAWKEYBOARD)]


class RAWINPUT(ctypes.Structure):
    _fields_ = [('header', RAWINPUTHEADER), ('data', _RAWDATA)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [('style', wintypes.UINT), ('lpfnWndProc', WNDPROC),
                ('cbClsExtra', ctypes.c_int), ('cbWndExtra', ctypes.c_int),
                ('hInstance', wintypes.HINSTANCE), ('hIcon', wintypes.HICON),
                ('hCursor', wintypes.HANDLE), ('hbrBackground', wintypes.HBRUSH),
                ('lpszMenuName', wintypes.LPCWSTR), ('lpszClassName', wintypes.LPCWSTR)]


class InputListener:
    """The player's physical keyboard and mouse, whichever window has focus.

    buttons maps names to ('key', virtual-key code) or ('mouse', 'left'|'right').
    take() returns, for the time since the last take(), the fraction each
    button was held and the raw mouse counts moved. Physical F9/F10 presses
    go to self.hotkeys; the time of the last physical mouse motion is kept in
    self.last_user_motion (play.py yields to the player).
    """

    def __init__(self, buttons):
        self.names = list(buttons)
        self._by_key = {code: k for k, (kind, code) in enumerate(buttons.values()) if kind == 'key'}
        self._by_mouse = {code: k for k, (kind, code) in enumerate(buttons.values()) if kind == 'mouse'}
        self._lock = threading.Lock()
        self._down_since = [None] * len(self.names)
        self._held = np.zeros(len(self.names))
        self._mouse = np.zeros(2)
        self._last_take = time.perf_counter()
        self.hotkeys = queue.Queue()
        self.last_user_motion = 0.0
        self.last_user_button = 0.0
        self.last_user_key = 0.0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name='raw-input', daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0) or self._error:
            raise RuntimeError(f'Raw input listener failed: {self._error}')

    _error = None

    def _set(self, k, down, now):
        since = self._down_since[k]
        if down and since is None:
            self._down_since[k] = now
        elif not down and since is not None:
            self._held[k] += now - max(since, self._last_take)
            self._down_since[k] = None

    def take(self):
        now = time.perf_counter()
        with self._lock:
            span = max(now - self._last_take, 1e-9)
            held = self._held.copy()
            for k, since in enumerate(self._down_since):
                if since is not None:
                    held[k] += now - max(since, self._last_take)
            mouse = self._mouse.copy()
            self._held[:] = 0.0
            self._mouse[:] = 0.0
            self._last_take = now
        return np.clip(held / span, 0.0, 1.0), mouse

    def held_now(self):
        with self._lock:
            return [s is not None for s in self._down_since]

    def _on_input(self, lparam):
        buf = RAWINPUT()
        size = wintypes.UINT(ctypes.sizeof(buf))
        if user32.GetRawInputData(lparam, RID_INPUT, ctypes.byref(buf), ctypes.byref(size),
                                  ctypes.sizeof(RAWINPUTHEADER)) in (0, 0xFFFFFFFF):
            return
        now = time.perf_counter()
        physical = bool(buf.header.hDevice)
        if buf.header.dwType == RIM_TYPEMOUSE:
            m = buf.data.mouse
            if not physical or m.ulExtraInformation == INJECTED_TAG:
                return
            with self._lock:
                if not m.usFlags & MOUSE_MOVE_ABSOLUTE and (m.lLastX or m.lLastY):
                    self._mouse += (m.lLastX, m.lLastY)
                    self.last_user_motion = now
                for flag, (button, down) in RI_MOUSE.items():
                    if m.usButtonFlags & flag and button in self._by_mouse:
                        self._set(self._by_mouse[button], down, now)
                        self.last_user_button = now
        elif buf.header.dwType == RIM_TYPEKEYBOARD:
            kb = buf.data.keyboard
            if not physical or kb.ExtraInformation == INJECTED_TAG:
                return
            down = not kb.Flags & RI_KEY_BREAK
            if down:
                self.last_user_key = now                 # any physical key, e.g. Esc to pause
            if down and kb.VKey in (VK_F9, VK_F10):
                self.hotkeys.put(kb.VKey)
            if kb.VKey in self._by_key:
                with self._lock:
                    self._set(self._by_key[kb.VKey], down, now)
                    self.last_user_button = now

    def _loop(self):
        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                try:
                    self._on_input(lparam)
                except Exception as e:           # never let an exception cross into Windows
                    print(f'raw input: {e}')
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc = WNDPROC(wndproc)        # keep a reference for the window's lifetime
        hinst = kernel32.GetModuleHandleW(None)
        cls = WNDCLASSW(lpfnWndProc=self._wndproc, hInstance=hinst,
                        lpszClassName=f'FlyDaggersRawInput{id(self)}')
        if not user32.RegisterClassW(ctypes.byref(cls)):
            self._error = f'RegisterClassW error {ctypes.get_last_error()}'
            self._ready.set()
            return
        hwnd = user32.CreateWindowExW(0, cls.lpszClassName, 'Fly Daggers input', 0,
                                      0, 0, 0, 0, None, None, hinst, None)
        devices = (RAWINPUTDEVICE * 2)(RAWINPUTDEVICE(1, 2, RIDEV_INPUTSINK, hwnd),   # mouse
                                       RAWINPUTDEVICE(1, 6, RIDEV_INPUTSINK, hwnd))   # keyboard
        if not hwnd or not user32.RegisterRawInputDevices(devices, 2, ctypes.sizeof(RAWINPUTDEVICE)):
            self._error = f'RegisterRawInputDevices error {ctypes.get_last_error()}'
            self._ready.set()
            return
        self._thread_id = kernel32.GetCurrentThreadId()
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.DestroyWindow(hwnd)

    def close(self):
        if getattr(self, '_thread_id', None):
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=2.0)


# ============================================================================
# The fly's input: SendInput
# ============================================================================

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE = 0x0001
MOUSE_FLAGS = {('left', True): 0x0002, ('left', False): 0x0004,
               ('right', True): 0x0008, ('right', False): 0x0010}
KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x0002, 0x0008


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('mouseData', wintypes.DWORD),
                ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
                ('dwExtraInfo', ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD), ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [('type', wintypes.DWORD), ('u', _INPUTUNION)]


def _send(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def _mouse_input(dx=0, dy=0, flags=MOUSEEVENTF_MOVE):
    return INPUT(type=INPUT_MOUSE, u=_INPUTUNION(mi=MOUSEINPUT(
        dx=int(dx), dy=int(dy), dwFlags=flags, dwExtraInfo=INJECTED_TAG)))


def _key_input(vk, down):
    scan = user32.MapVirtualKeyW(vk, 0)
    flags = KEYEVENTF_SCANCODE | (0 if down else KEYEVENTF_KEYUP)
    return INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(
        wVk=0, wScan=scan, dwFlags=flags, dwExtraInfo=INJECTED_TAG)))


class Controller:
    """Holds the fly's buttons and moves its mouse at a steady velocity.

    A background thread spreads each look's mouse motion over the look in
    small steps, as a hand would, instead of one jump per look.
    """

    STEP_S = 0.005

    def __init__(self, buttons):
        self.buttons = dict(buttons)
        self.down = set()
        self._velocity = np.zeros(2)          # counts per second
        self._carry = np.zeros(2)
        self._enabled = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._mover, name='fly-mouse', daemon=True)
        self._thread.start()

    def _press(self, name, down):
        kind, code = self.buttons[name]
        if kind == 'key':
            _send(_key_input(code, down))
        else:
            _send(_mouse_input(flags=MOUSE_FLAGS[code, down]))
        (self.down.add if down else self.down.discard)(name)

    def tap(self, vk, hold_s=0.06):
        """Press and release one key (virtual-key code), e.g. R to restart a run."""
        _send(_key_input(vk, True))
        time.sleep(hold_s)
        _send(_key_input(vk, False))

    def apply(self, held, velocity):
        """held: {name: bool}; velocity: mouse [x, y] counts per second."""
        for name, want in held.items():
            if want != (name in self.down):
                self._press(name, want)
        with self._lock:
            self._velocity = np.asarray(velocity, dtype=float)
            self._enabled = True

    def release_all(self):
        with self._lock:
            self._enabled = False
            self._velocity[:] = 0.0
            self._carry[:] = 0.0
        for name in list(self.down):
            self._press(name, False)

    def _mover(self):
        last = time.perf_counter()
        while not self._stop.wait(self.STEP_S):
            now = time.perf_counter()
            with self._lock:
                if not self._enabled:
                    last = now
                    continue
                self._carry += self._velocity * (now - last)
                move = np.trunc(self._carry)
                self._carry -= move
            last = now
            if move.any():
                _send(_mouse_input(*move))

    def close(self):
        self.release_all()
        self._stop.set()
        self._thread.join(timeout=1.0)
