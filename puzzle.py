from tkinter import (
    Frame, Label, CENTER, Canvas,
    StringVar, BooleanVar, Radiobutton, Checkbutton,
    Button, Toplevel
)
import random
import logic
import constants as c
from agent_Minimax import MinimaxAgent
from agent_Expectimax import ExpectimaxAgent


def _clone(mat):
    return [row[:] for row in mat]


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        # ===== window / layout =====
        self.master.title("2048 (Modified)")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=0)  # topbar
        self.grid_rowconfigure(1, weight=1)  # board area
        self.grid_columnconfigure(0, weight=1)

        # ---- safe defaults if constants.py doesn't have these ----
        self.AI_DEPTH = getattr(c, "AI_DEPTH", 3)
        self.AI_DELAY_MS = getattr(c, "AI_DELAY_MS", 180)
        self.KEY_TOGGLE_CONTROLLER = getattr(c, "KEY_TOGGLE_CONTROLLER", "m")
        self.KEY_AI_STEP = getattr(c, "KEY_AI_STEP", "space")
        self.KEY_RESTART = getattr(c, "KEY_RESTART", "r")
        self.KEY_TOGGLE_AI_TYPE = getattr(c, "KEY_TOGGLE_AI_TYPE", "t")  # 可选：键盘切换 AI Type

        # ===== controller state =====
        # Controller: Human / AI
        self.controller_var = StringVar(value="Human")   # "Human" or "AI"
        self.autoplay_var = BooleanVar(value=True)

        # AI Type: Minimax / Expectimax
        self.ai_type_var = StringVar(value="Minimax")    # "Minimax" or "Expectimax"

        self.last_move = "-"
        self.step_count = 0
        self.ai_job = None
        self.end_popup = None

        # ===== game state =====
        self.special_cell_pos = self._random_special_pos()
        self.agent = self._build_agent()  # uses ai_type_var

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        # ===== UI =====
        self._init_topbar()
        self._init_board_canvas()

        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = [_clone(self.matrix)]

        # canvas draw state
        self._resize_job = None
        self.board_size = None
        self.pad = None
        self.cell_size = None
        self.font_size = None
        self.rect_ids = [[None for _ in range(c.GRID_LEN)] for _ in range(c.GRID_LEN)]
        self.text_ids = [[None for _ in range(c.GRID_LEN)] for _ in range(c.GRID_LEN)]

        self._ensure_canvas_items_created()
        self._update_status()
        self.update_grid_cells()

        # ===== events =====
        self.master.bind("<Key>", self.key_down)
        self.master.bind("<Configure>", self._on_configure)  # adaptive resizing

        # minsize (allow enlarge)
        self.master.update_idletasks()
        w = self.master.winfo_reqwidth()
        h = self.master.winfo_reqheight()
        self.master.minsize(w, h)
        self.master.resizable(True, True)

        self._sync_ai_loop()
        self.mainloop()

    # ---------------- Agent factory ----------------

    def _build_agent(self):
        """根据 ai_type_var 构造 agent，并把特殊格子位置注入。"""
        ai_type = self.ai_type_var.get()
        if ai_type == "Expectimax":
            return ExpectimaxAgent(depth=self.AI_DEPTH, special_pos=self.special_cell_pos)
        return MinimaxAgent(depth=self.AI_DEPTH, special_pos=self.special_cell_pos)

    def _on_ai_type_change(self):
        """UI 切换 AI Type 后：重建 agent，并同步 auto loop。"""
        self._stop_ai_loop()
        self.agent = self._build_agent()
        self._update_status()
        self._sync_ai_loop()

    # ---------------- Topbar ----------------

    def _init_topbar(self):
        self.topbar = Frame(self)
        self.topbar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self.topbar.grid_columnconfigure(6, weight=1)

        # Controller: Human / AI
        Radiobutton(
            self.topbar, text="Human", variable=self.controller_var, value="Human",
            command=self._on_controller_change
        ).grid(row=0, column=0, padx=6)

        Radiobutton(
            self.topbar, text="AI", variable=self.controller_var, value="AI",
            command=self._on_controller_change
        ).grid(row=0, column=1, padx=6)

        # AI Type: Minimax / Expectimax
        Label(self.topbar, text="AI Type:").grid(row=0, column=2, padx=(14, 4))

        Radiobutton(
            self.topbar, text="Minimax", variable=self.ai_type_var, value="Minimax",
            command=self._on_ai_type_change
        ).grid(row=0, column=3, padx=4)

        Radiobutton(
            self.topbar, text="Expectimax", variable=self.ai_type_var, value="Expectimax",
            command=self._on_ai_type_change
        ).grid(row=0, column=4, padx=4)

        # Auto-play
        Checkbutton(
            self.topbar, text="Auto-play", variable=self.autoplay_var,
            command=self._on_autoplay_change
        ).grid(row=0, column=5, padx=10)

        self.status_label = Label(self.topbar, text="", anchor="w", justify="left")
        self.status_label.grid(row=0, column=6, padx=10, sticky="ew")

    # ---------------- Canvas board (adaptive) ----------------

    def _init_board_canvas(self):
        self.board_container = Frame(self)
        self.board_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.board_container.grid_rowconfigure(0, weight=1)
        self.board_container.grid_columnconfigure(0, weight=1)

        self.canvas = Canvas(self.board_container, bg="#ffffff", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

    def _on_configure(self, event):
        """Debounce resize to avoid heavy redraw on every pixel."""
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(40, self._apply_resize)

    def _apply_resize(self):
        self._resize_job = None
        self._recompute_board_geometry()
        self._layout_canvas_items()
        self.update_grid_cells()

        # topic wrap
        try:
            w = self.master.winfo_width()
            self.status_label.configure(wraplength=max(320, w - 420))
        except Exception:
            pass

    def _recompute_board_geometry(self):
        """Compute board_size/padding/cell_size/font_size based on current window size."""
        cw = max(1, self.board_container.winfo_width())
        ch = max(1, self.board_container.winfo_height())

        margin = 10
        size = max(220, min(cw, ch) - margin)
        self.board_size = size

        base_size = getattr(c, "SIZE", 500) or 500
        base_pad = getattr(c, "GRID_PADDING", 10) or 10
        self.pad = max(4, int(self.board_size * base_pad / base_size))

        usable = self.board_size - self.pad * (c.GRID_LEN + 1)
        self.cell_size = max(10, usable // c.GRID_LEN)

        self.font_size = max(10, int(self.cell_size * 0.35))

    def _ensure_canvas_items_created(self):
        """Create rectangles and texts once."""
        self._recompute_board_geometry()

        self.canvas.delete("all")

        # board background rect
        self.canvas.create_rectangle(
            0, 0, self.board_size, self.board_size,
            fill=c.BACKGROUND_COLOR_GAME,
            outline=c.BACKGROUND_COLOR_GAME,
            tags=("board_bg",)
        )

        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                r = self.canvas.create_rectangle(0, 0, 0, 0, fill=c.BACKGROUND_COLOR_CELL_EMPTY, outline="")
                t = self.canvas.create_text(0, 0, text="", font=("Verdana", self.font_size, "bold"))
                self.rect_ids[i][j] = r
                self.text_ids[i][j] = t

        self._layout_canvas_items()

    def _layout_canvas_items(self):
        """Update coords of all rectangles/texts based on latest board geometry."""
        if self.board_size is None:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        dx = max(0, (cw - self.board_size) // 2)
        dy = max(0, (ch - self.board_size) // 2)

        self.canvas.coords("board_bg", dx, dy, dx + self.board_size, dy + self.board_size)

        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                x0 = dx + self.pad + j * (self.cell_size + self.pad)
                y0 = dy + self.pad + i * (self.cell_size + self.pad)
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                self.canvas.coords(self.rect_ids[i][j], x0, y0, x1, y1)
                self.canvas.coords(self.text_ids[i][j], (x0 + x1) / 2, (y0 + y1) / 2)
                self.canvas.itemconfigure(self.text_ids[i][j], font=("Verdana", self.font_size, "bold"))

    # ---------------- Game rules (modified) ----------------

    def _random_special_pos(self):
        center_start = (c.GRID_LEN - 2) // 2
        center_end = center_start + 2
        return (
            random.randint(center_start, center_end - 1),
            random.randint(center_start, center_end - 1)
        )

    def _apply_special_effect(self):
        """改版规则：成功移动后，特殊格子上的 tile 值 >2 则减半。"""
        i, j = self.special_cell_pos
        if self.matrix[i][j] > 2:
            self.matrix[i][j] //= 2

    # ---------------- Rendering ----------------

    def update_grid_cells(self):
        """Update canvas colors/texts to reflect matrix."""
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                v = self.matrix[i][j]
                is_special = (i, j) == self.special_cell_pos

                if v == 0:
                    bg = c.BACKGROUND_COLOR_SPECIAL_CELL if is_special else c.BACKGROUND_COLOR_CELL_EMPTY
                    self.canvas.itemconfigure(self.rect_ids[i][j], fill=bg)
                    self.canvas.itemconfigure(self.text_ids[i][j], text="")
                else:
                    if is_special:
                        bg = c.BACKGROUND_COLOR_SPECIAL_CELL
                        fg = c.CELL_COLOR_DICT.get(v, "#776e65")
                    else:
                        bg = c.BACKGROUND_COLOR_DICT.get(v, c.BACKGROUND_COLOR_CELL_EMPTY)
                        fg = c.CELL_COLOR_DICT.get(v, "#f9f6f2")

                    self.canvas.itemconfigure(self.rect_ids[i][j], fill=bg)
                    self.canvas.itemconfigure(self.text_ids[i][j], text=str(v), fill=fg)

        self.canvas.update_idletasks()

    def _update_status(self):
        controller = self.controller_var.get()
        auto = self.autoplay_var.get()
        ai_type = self.ai_type_var.get()
        state = logic.game_state(self.matrix)

        if controller == "Human":
            msg = (
                f"Controller: Human | Steps: {self.step_count} | Last: {self.last_move} | "
                f"State: {state} ({self.KEY_TOGGLE_CONTROLLER}=Toggle, {self.KEY_RESTART}=Restart)"
            )
        else:
            msg = (
                f"Controller: AI ({ai_type}, depth={self.AI_DEPTH}) | Auto: {auto} | "
                f"Steps: {self.step_count} | Last: {self.last_move} | State: {state} "
                f"(Space=Step, {self.KEY_TOGGLE_CONTROLLER}=Toggle, {self.KEY_TOGGLE_AI_TYPE}=AIType, {self.KEY_RESTART}=Restart)"
            )
        self.status_label.configure(text=msg)

    # ---------------- End popup ----------------

    def _show_end_popup(self, result):
        if self.end_popup is not None:
            return

        self._stop_ai_loop()
        mode = self.controller_var.get()
        ai_type = self.ai_type_var.get()
        steps = self.step_count

        if result == "win":
            bg = "#1B5E20"
            fg = "#FFD54F"
            title = "🎉 YOU WIN!"
        else:
            bg = "#424242"
            fg = "#FFFFFF"
            title = "Game Over"

        top = Toplevel(self.master)
        self.end_popup = top
        top.title("Result")
        top.configure(bg=bg)
        top.resizable(False, False)
        top.transient(self.master)
        top.grab_set()

        Label(top, text=title, font=("Verdana", 20, "bold"), bg=bg, fg=fg).pack(padx=18, pady=(18, 8))

        if mode == "AI":
            info = f"Mode: AI ({ai_type})    Steps: {steps}"
        else:
            info = f"Mode: Human    Steps: {steps}"

        Label(top, text=info, font=("Verdana", 12, "bold"), bg=bg, fg=fg).pack(padx=18, pady=(0, 14))

        btn_row = Frame(top, bg=bg)
        btn_row.pack(pady=(0, 16))
        Button(btn_row, text="Restart", command=self.restart).grid(row=0, column=0, padx=8)
        Button(btn_row, text="OK", command=self._close_end_popup).grid(row=0, column=1, padx=8)

    def _close_end_popup(self):
        if self.end_popup is None:
            return
        try:
            self.end_popup.grab_release()
        except Exception:
            pass
        try:
            self.end_popup.destroy()
        except Exception:
            pass
        self.end_popup = None

    # ---------------- restart ----------------

    def restart(self):
        self._stop_ai_loop()
        self._close_end_popup()

        self.step_count = 0
        self.last_move = "RESTART"

        self.special_cell_pos = self._random_special_pos()

        # rebuild agent to ensure special_pos is aligned and ai_type is honored
        self.agent = self._build_agent()

        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = [_clone(self.matrix)]

        self._update_status()
        self.update_grid_cells()
        self._sync_ai_loop()

    # ---------------- move wrappers ----------------

    def _post_move_updates(self):
        self._apply_special_effect()
        self.matrix = logic.add_two(self.matrix)

        self.history_matrixs.append(_clone(self.matrix))
        self.update_grid_cells()

        state = logic.game_state(self.matrix)
        if state == "win":
            if self.controller_var.get() == "AI":
                print(f"[{self.ai_type_var.get()}] WIN in {self.step_count} steps.")
            self._show_end_popup("win")
        elif state == "lose":
            self._show_end_popup("lose")

    def _try_move_by_func(self, move_fn, move_name):
        if self.end_popup is not None:
            return False

        new_mat, done = move_fn(_clone(self.matrix))
        if not done:
            return False

        self.matrix = new_mat
        self.last_move = move_name
        self.step_count += 1

        self._post_move_updates()
        self._update_status()
        return True

    def _try_move_by_name(self, move_name):
        mapping = {"Up": logic.up, "Down": logic.down, "Left": logic.left, "Right": logic.right}
        fn = mapping.get(move_name)
        if fn is None:
            return False
        return self._try_move_by_func(fn, move_name)

    # ---------------- Controller / AI loop ----------------

    def _on_controller_change(self):
        self._update_status()
        self._sync_ai_loop()

    def _on_autoplay_change(self):
        self._update_status()
        self._sync_ai_loop()

    def _sync_ai_loop(self):
        self._stop_ai_loop()
        if self.controller_var.get() == "AI" and self.autoplay_var.get() and self.end_popup is None:
            self._start_ai_loop()

    def _start_ai_loop(self):
        self._stop_ai_loop()
        self.ai_job = self.after(self.AI_DELAY_MS, self._ai_tick)

    def _stop_ai_loop(self):
        if self.ai_job is not None:
            try:
                self.after_cancel(self.ai_job)
            except Exception:
                pass
            self.ai_job = None

    def _ai_tick(self):
        if self.controller_var.get() != "AI" or not self.autoplay_var.get() or self.end_popup is not None:
            self.ai_job = None
            return
        if logic.game_state(self.matrix) != "not over":
            self.ai_job = None
            self._update_status()
            return

        move = self.agent.choose_move(self.matrix)
        if move is None:
            self.ai_job = None
            self._update_status()
            return

        self._try_move_by_name(move)
        self.ai_job = self.after(self.AI_DELAY_MS, self._ai_tick)

    def _ai_step_once(self):
        if self.controller_var.get() != "AI" or self.end_popup is not None:
            return
        if logic.game_state(self.matrix) != "not over":
            return

        move = self.agent.choose_move(self.matrix)
        if move is None:
            self._update_status()
            return
        self._try_move_by_name(move)

    # ---------------- Input handler ----------------

    def key_down(self, event):
        key = event.keysym

        if key == c.KEY_QUIT:
            exit()

        if key == self.KEY_RESTART:
            self.restart()
            return

        # toggle controller (Human <-> AI)
        if key == self.KEY_TOGGLE_CONTROLLER:
            self.controller_var.set("AI" if self.controller_var.get() == "Human" else "Human")
            self._on_controller_change()
            return

        # toggle AI Type (Minimax <-> Expectimax)
        if key == self.KEY_TOGGLE_AI_TYPE:
            self.ai_type_var.set("Expectimax" if self.ai_type_var.get() == "Minimax" else "Minimax")
            self._on_ai_type_change()
            return

        # undo
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.history_matrixs.pop()
            self.matrix = _clone(self.history_matrixs[-1])
            if self.step_count > 0:
                self.step_count -= 1
            self.last_move = "UNDO"
            self._update_status()
            self.update_grid_cells()
            return

        # AI step (only when AI and not autoplay)
        if key == self.KEY_AI_STEP and self.controller_var.get() == "AI" and not self.autoplay_var.get():
            self._ai_step_once()
            return

        # Human mode: handle move keys
        if self.controller_var.get() == "Human":
            if key in self.commands:
                self._try_move_by_func(self.commands[key], key)
            return


game_grid = GameGrid()
