from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c

def gen():
    return random.randint(0, c.GRID_LEN - 1)

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

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

        self.grid_cells = []
        # 特殊格子位置（随机生成在中间2x2的四个格子中）
        # 对于5x5网格，中间2x2区域的行列索引是 1, 2
        center_start = (c.GRID_LEN - 2) // 2  # 计算中间区域的起始索引
        center_end = center_start + 2
        self.special_cell_pos = (
            random.randint(center_start, center_end - 1),
            random.randint(center_start, center_end - 1)
        )
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.update_grid_cells()

        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                # 检查是否是特殊格子
                is_special = (i, j) == self.special_cell_pos
                cell_bg = c.BACKGROUND_COLOR_SPECIAL_CELL if is_special else c.BACKGROUND_COLOR_CELL_EMPTY
                
                cell = Frame(
                    background,
                    bg=cell_bg,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=cell_bg,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                is_special = (i, j) == self.special_cell_pos
                
                if new_number == 0:
                    # 如果是特殊格子，保持特殊颜色；否则使用默认颜色
                    bg_color = c.BACKGROUND_COLOR_SPECIAL_CELL if is_special else c.BACKGROUND_COLOR_CELL_EMPTY
                    self.grid_cells[i][j].configure(text="", bg=bg_color)
                else:
                    # 特殊格子上的tile显示时，背景色保持特殊格子的颜色
                    if is_special:
                        self.grid_cells[i][j].configure(
                            text=str(new_number),
                            bg=c.BACKGROUND_COLOR_SPECIAL_CELL,
                            fg=c.CELL_COLOR_DICT.get(new_number, "#776e65")
                        )
                    else:
                        self.grid_cells[i][j].configure(
                            text=str(new_number),
                            bg=c.BACKGROUND_COLOR_DICT[new_number],
                            fg=c.CELL_COLOR_DICT[new_number]
                        )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT: exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                # 应用特殊格子的效果：在特殊格子上的tile数值除以2（如果大于2）
                special_i, special_j = self.special_cell_pos
                if self.matrix[special_i][special_j] > 2:
                    self.matrix[special_i][special_j] //= 2
                
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

game_grid = GameGrid()  