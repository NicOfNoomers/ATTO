"""Panel management for the pump dashboard (Tkinter).

* ScrollDock: a sidebar that stacks panels vertically and scrolls when they
  don't fit (mouse wheel while the pointer is over it).
* PanelShell: a panel with a header bar: click the title to collapse/expand,
  the pop-out button turns the panel into its own window (Tk "wm manage",
  e.g. for a second screen; closing that window docks it back), the cross
  hides it (View menu brings it back).
* PanelManager: the registry, the View menu (show/hide each panel, presets,
  collapse/expand all, text size, reset), and the layout file. The layout
  (window geometry, sidebar widths, each panel's visible/collapsed/popped
  state, text size) is saved per PC in %LOCALAPPDATA%/ATTO, not in the
  configs, and restored at start.

Tk can't move a widget to another parent, so each panel lives in the dock it
was built in; it can be hidden, collapsed or popped out from there.
"""
from __future__ import annotations

import json
import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Callable, Dict, List, Optional

HEADER_BG = "#dfe3ea"
HEADER_BG_HOVER = "#cfd5df"
LAYOUT_VERSION = 1
FONT_SIZES = {"Small": 8, "Normal": 9, "Large": 11, "Larger": 13}
NAMED_FONTS = ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont",
               "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont", "TkFixedFont")


def default_layout_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ATTO", "dashboard_layout.json")


class ScrollDock(ttk.Frame):
    """Vertical stack of panels that scrolls when it is taller than the window."""

    def __init__(self, master, name: str, **kw):
        super().__init__(master, **kw)
        self.name = name
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind("<Enter>", lambda _e: self._bind_wheel(True))
        self.bind("<Leave>", lambda _e: self._bind_wheel(False))
        self.shells: List["PanelShell"] = []
        self.on_repack: Optional[Callable[[], None]] = None

    def _on_scroll_set(self, lo, hi):
        # Scrollbar only when needed
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.vbar.pack_forget()
        elif not self.vbar.winfo_ismapped():
            self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vbar.set(lo, hi)

    def _on_inner_configure(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)

    def _bind_wheel(self, on: bool):
        if on:
            self.bind_all("<MouseWheel>", self._on_wheel, add="+")
        else:
            self.unbind_all("<MouseWheel>")

    def _on_wheel(self, e):
        # Only scroll if the pointer is over this dock and it can scroll
        w = self.winfo_containing(e.x_root, e.y_root)
        while w is not None and w is not self:
            w = w.master
        if w is self and self.vbar.winfo_ismapped():
            self.canvas.yview_scroll(int(-e.delta / 120), "units")

    def repack(self):
        for s in self.shells:
            s.pack_forget()
        for s in self.shells:
            if s.visible and not s.popped:
                s.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(0, 4))
        self.after_idle(self._on_inner_configure)
        if self.on_repack:
            self.on_repack()


class PanelShell(tk.Frame):
    """A panel with a header (collapse / pop out / hide) and a body frame."""

    def __init__(self, dock: ScrollDock, key: str, title: str, manager: "PanelManager"):
        super().__init__(dock.inner, bd=1, relief=tk.GROOVE)
        self.key, self.title, self.dock, self.manager = key, title, dock, manager
        self.visible, self.collapsed, self.popped = True, False, False
        self.popped_geometry: Optional[str] = None
        self.header = tk.Frame(self, bg=HEADER_BG, cursor="hand2")
        self.header.pack(side=tk.TOP, fill=tk.X)
        self.arrow = tk.Label(self.header, text="▾", bg=HEADER_BG, width=2)
        self.arrow.pack(side=tk.LEFT)
        self.lbl = tk.Label(self.header, text=title, bg=HEADER_BG, anchor="w",
                            font=tkfont.nametofont("TkHeadingFont"))
        self.lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_close = tk.Label(self.header, text="✕", bg=HEADER_BG, padx=6)
        self.btn_close.pack(side=tk.RIGHT)
        self.btn_pop = tk.Label(self.header, text="⧉", bg=HEADER_BG, padx=6)
        self.btn_pop.pack(side=tk.RIGHT)
        for w in (self.header, self.arrow, self.lbl):
            w.bind("<Button-1>", lambda _e: self.toggle_collapsed())
            w.bind("<Enter>", lambda _e: self._hover(True))
            w.bind("<Leave>", lambda _e: self._hover(False))
        self.btn_pop.bind("<Button-1>", lambda _e: self.toggle_popped())
        self.btn_close.bind("<Button-1>", lambda _e: self.manager.set_visible(self.key, False))
        self.body = ttk.Frame(self)
        self.body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        dock.shells.append(self)

    def _hover(self, on: bool):
        bg = HEADER_BG_HOVER if on else HEADER_BG
        for w in (self.header, self.arrow, self.lbl, self.btn_pop, self.btn_close):
            w.configure(bg=bg)

    def set_collapsed(self, collapsed: bool):
        self.collapsed = bool(collapsed) and not self.popped
        if self.collapsed:
            self.body.pack_forget()
        elif not self.body.winfo_ismapped():
            self.body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.arrow.configure(text="▸" if self.collapsed else "▾")
        self.dock.after_idle(self.dock._on_inner_configure)

    def toggle_collapsed(self):
        self.set_collapsed(not self.collapsed)

    def set_popped(self, popped: bool):
        if popped == self.popped:
            return
        if popped:
            self.set_collapsed(False)
            self.pack_forget()
            self.tk.call("wm", "manage", self)
            self.tk.call("wm", "title", self, self.title)
            self.tk.call("wm", "protocol", self, "WM_DELETE_WINDOW", self.register(lambda: self.set_popped(False)))
            if self.popped_geometry:
                try:
                    self.tk.call("wm", "geometry", self, self.popped_geometry)
                except tk.TclError:
                    pass
            self.header.pack_forget()       # the window's title bar replaces the header
            self.popped = True
        else:
            try:
                self.popped_geometry = str(self.tk.call("wm", "geometry", self))
            except tk.TclError:
                pass
            self.tk.call("wm", "forget", self)
            self.body.pack_forget()
            self.header.pack(side=tk.TOP, fill=tk.X)
            self.body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
            self.popped = False
        self.dock.repack()

    def toggle_popped(self):
        self.set_popped(not self.popped)

    def state(self) -> Dict:
        geo = self.popped_geometry
        if self.popped:
            try:
                geo = str(self.tk.call("wm", "geometry", self))
            except tk.TclError:
                pass
        return {"visible": self.visible, "collapsed": self.collapsed, "popped": self.popped, "popped_geometry": geo}


class PanelManager:
    def __init__(self, root: tk.Tk, layout_path: Optional[str] = None):
        self.root = root
        self.layout_path = layout_path or default_layout_path()
        self.panels: Dict[str, PanelShell] = {}
        self.docks: Dict[str, ScrollDock] = {}
        self.presets: Dict[str, Dict[str, str]] = {}
        self.vis_vars: Dict[str, tk.BooleanVar] = {}
        self.font_size = tk.StringVar(value="Normal")
        self.paned: Optional[ttk.PanedWindow] = None
        self._saved: Dict = {}
        self._columns: List = []

    # ---- building ----
    def add_dock(self, name: str, master) -> ScrollDock:
        d = ScrollDock(master, name)
        d.on_repack = self._update_columns
        self.docks[name] = d
        return d

    def set_columns(self, columns: List):
        """The paned window's children in order; sidebars (ScrollDocks) with no
        visible docked panel are taken out so the plots get the width."""
        self._columns = list(columns)

    def _update_columns(self):
        if self.paned is None or not self._columns:
            return
        want = [c for c in self._columns
                if not isinstance(c, ScrollDock) or any(s.visible and not s.popped for s in c.shells)]
        current = [str(p) for p in self.paned.panes()]
        if [str(c) for c in want] == current:
            return
        for c in self._columns:
            if str(c) in current:
                self.paned.forget(c)
        for c in want:
            self.paned.add(c, weight=0 if isinstance(c, ScrollDock) else 1)

    def add_panel(self, key: str, title: str, dock: str, builder: Callable[[ttk.Frame], None]) -> PanelShell:
        shell = PanelShell(self.docks[dock], key, title, self)
        builder(shell.body)
        self.panels[key] = shell
        self.vis_vars[key] = tk.BooleanVar(value=True)
        self.docks[dock].repack()
        return shell

    def add_preset(self, name: str, states: Dict[str, str]):
        """states: panel key -> "open" | "collapsed" | "hidden" (others: collapsed)."""
        self.presets[name] = states

    def build_view_menu(self, menubar: tk.Menu) -> tk.Menu:
        m = tk.Menu(menubar, tearoff=False)
        for key, shell in self.panels.items():
            m.add_checkbutton(label=shell.title, variable=self.vis_vars[key],
                              command=lambda k=key: self.set_visible(k, self.vis_vars[k].get()))
        m.add_separator()
        pm = tk.Menu(m, tearoff=False)
        for name in self.presets:
            pm.add_command(label=name, command=lambda n=name: self.apply_preset(n))
        m.add_cascade(label="Layout preset", menu=pm)
        m.add_command(label="Expand all", command=lambda: self._all_collapsed(False))
        m.add_command(label="Collapse all", command=lambda: self._all_collapsed(True))
        m.add_command(label="Dock all pop-out windows", command=self.dock_all)
        m.add_separator()
        fm = tk.Menu(m, tearoff=False)
        for name in FONT_SIZES:
            fm.add_radiobutton(label=name, value=name, variable=self.font_size,
                               command=lambda: self.set_font_size(self.font_size.get()))
        m.add_cascade(label="Text size", menu=fm)
        m.add_separator()
        m.add_command(label="Save layout now", command=self.save)
        m.add_command(label="Reset layout", command=self.reset)
        menubar.add_cascade(label="View", menu=m)
        return m

    # ---- state changes ----
    def set_visible(self, key: str, visible: bool):
        shell = self.panels[key]
        if not visible and shell.popped:
            shell.set_popped(False)
        shell.visible = bool(visible)
        self.vis_vars[key].set(shell.visible)
        shell.dock.repack()

    def _all_collapsed(self, collapsed: bool):
        for s in self.panels.values():
            s.set_collapsed(collapsed)

    def dock_all(self):
        for s in self.panels.values():
            s.set_popped(False)

    def apply_preset(self, name: str):
        states = self.presets.get(name, {})
        for key, shell in self.panels.items():
            st = states.get(key, "collapsed")
            if shell.popped and st == "hidden":
                shell.set_popped(False)
            shell.visible = st != "hidden"
            self.vis_vars[key].set(shell.visible)
            if not shell.popped:
                shell.set_collapsed(st == "collapsed")
        for d in self.docks.values():
            d.repack()

    def set_font_size(self, name: str):
        size = FONT_SIZES.get(name, 9)
        self.font_size.set(name if name in FONT_SIZES else "Normal")
        for f in NAMED_FONTS:
            try:
                tkfont.nametofont(f).configure(size=size)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont("TkHeadingFont").configure(size=size, weight="bold")
        except tk.TclError:
            pass

    # ---- persistence ----
    def load(self) -> Dict:
        try:
            with open(self.layout_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == LAYOUT_VERSION:
                self._saved = data
        except Exception:
            self._saved = {}
        return self._saved

    def restore(self, default_preset: Optional[str] = None):
        """Apply the saved layout (or the default preset when there is none)."""
        data = self._saved
        if not data:
            if default_preset:
                self.apply_preset(default_preset)
            return
        self.set_font_size(data.get("font_size", "Normal"))
        for key, st in (data.get("panels") or {}).items():
            shell = self.panels.get(key)
            if not shell:
                continue
            shell.visible = bool(st.get("visible", True))
            self.vis_vars[key].set(shell.visible)
            shell.popped_geometry = st.get("popped_geometry")
            shell.set_collapsed(bool(st.get("collapsed", False)))
        for d in self.docks.values():
            d.repack()
        # pop-outs and sash positions once the window has its size
        self.root.after(300, self._restore_late, data)

    def _restore_late(self, data):
        for key, st in (data.get("panels") or {}).items():
            shell = self.panels.get(key)
            if shell and st.get("popped") and shell.visible:
                shell.set_popped(True)
        sash = data.get("sash")
        if self.paned is not None and isinstance(sash, list):
            for i, x in enumerate(sash):
                try:
                    self.paned.sashpos(i, int(x))
                except tk.TclError:
                    pass

    def window_geometry(self) -> Optional[str]:
        """Saved main-window geometry if it still fits on this screen."""
        geo = self._saved.get("geometry")
        try:
            size, x, y = geo.split("+")[0], int(geo.split("+")[1]), int(geo.split("+")[2])
            w, h = (int(v) for v in size.split("x"))
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            if 0 <= x < sw - 100 and 0 <= y < sh - 100 and w >= 600 and h >= 400:
                return geo
        except Exception:
            pass
        return None

    def save(self):
        data = {"version": LAYOUT_VERSION, "geometry": self.root.winfo_geometry(),
                "font_size": self.font_size.get(),
                "panels": {k: s.state() for k, s in self.panels.items()}}
        if self.paned is not None:
            try:
                data["sash"] = [self.paned.sashpos(i) for i in range(len(self.paned.panes()) - 1)]
            except tk.TclError:
                pass
        try:
            os.makedirs(os.path.dirname(self.layout_path), exist_ok=True)
            tmp = self.layout_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.layout_path)
        except Exception:
            pass
        return data

    def reset(self):
        self.dock_all()
        self.set_font_size("Normal")
        if self.presets:
            self.apply_preset(next(iter(self.presets)))
        try:
            os.remove(self.layout_path)
        except OSError:
            pass
