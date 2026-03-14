from dataclasses import dataclass

@dataclass(frozen=True)
class LayoutConfig:
    grid_size: int = 4
    padding: int = 10
    gap: int = 10
    cell_size: int = 70
    sidebar_width: int = 250

    color_map = {
        0: ("#cdc1b4", "#776e65"),  # 空单元格
        1: ("#eee4da", "#776e65"),  # 2 - 浅米色
        2: ("#ede0c8", "#776e65"),  # 4 - 浅卡其色
        3: ("#f2b179", "#f9f6f2"),  # 8 - 浅橙色（文字变白）
        4: ("#f59563", "#f9f6f2"),  # 16 - 橙红色
        5: ("#f67c5f", "#f9f6f2"),  # 32 - 深橙色
        6: ("#f65e3b", "#f9f6f2"),  # 64 - 红色
        7: ("#edcf72", "#f9f6f2"),  # 128 - 浅黄色
        8: ("#edcc61", "#f9f6f2"),  # 256 - 金色
        9: ("#edc850", "#f9f6f2"),  # 512 - 深金色
        10: ("#edc53f", "#f9f6f2"),  # 1024 - 亮金色
        11: ("#edc22e", "#f9f6f2"),  # 2048 - 核心黄色
        12: ("#3c3a32", "#f9f6f2"),  # 4096 - 深灰色
        13: ("#3c3a32", "#f9f6f2")  # 8192 - 深灰色（更高数字通用）
    }

    @property
    def canvas_width(self) -> float:
        # 2*padding + (grid_size-1)*gap + grid_size*cell_size + sidebar
        return 2 * self.padding + (self.grid_size - 1) * self.gap + self.grid_size * self.cell_size + self.sidebar_width

    @property
    def canvas_height(self) -> float:
        return 2 * self.padding + (self.grid_size - 1) * self.gap + self.grid_size * self.cell_size