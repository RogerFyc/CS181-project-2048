import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# 1) 录入你提供的“无限步数”数据
# ------------------------
df = pd.DataFrame({
    "Agent": ["Minimax", "Expectimax", "Actor-Critic", "DQN"],

    # 1) Win/Loss/Stuck
    "Wins": [17, 13, 5, 0],
    "Losses": [0, 0, 11, 5],
    "Stuck": [3, 7, 4, 15],
    "WinRate_pct": [85.0, 65.0, 25.0, 0.0],

    # 2) Max tile achievement
    "AvgMaxTile": [233.6, 198.4, 150.4, 50.4],
    "BestMaxTile": [256, 256, 256, 128],
    "Rate256_pct": [85.0, 65.0, 25.0, 0.0],

    # 3) Step efficiency
    "AvgTotalSteps": [314.1, 452.2, 315.1, 511.8],
    "AvgWinSteps": [265.0, 358.4, 262.0, np.nan],
    "FastestWinSteps": [148, 191, 191, np.nan],

    # 4) Outcome distribution (%)
    "OutcomeWin_pct": [85.0, 65.0, 25.0, 0.0],
    "OutcomeStuck_pct": [15.0, 35.0, 20.0, 75.0],
    "OutcomeFail_pct": [0.0, 0.0, 55.0, 25.0],
})

# ------------------------
# 2) 四张单独图
# ------------------------

# 图1：胜率
plt.figure(figsize=(7, 4))
plt.bar(df["Agent"], df["WinRate_pct"])
plt.title("Win Rate (Unlimited steps)")
plt.ylabel("Win Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 图2：最大数字成就（AvgMaxTile + 标注 256 比例）
plt.figure(figsize=(7, 4))
plt.bar(df["Agent"], df["AvgMaxTile"])
plt.title("Average Max Tile (Unlimited steps)")
plt.ylabel("Avg Max Tile")
plt.xticks(rotation=20)
for i, r in df.iterrows():
    plt.text(i, r["AvgMaxTile"], f'256 rate: {r["Rate256_pct"]:.0f}%',
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# 图3：步数效率（AvgTotalSteps）
plt.figure(figsize=(7, 4))
plt.bar(df["Agent"], df["AvgTotalSteps"])
plt.title("Average Total Steps (Unlimited steps)")
plt.ylabel("Avg Total Steps")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 图4：结果分布（堆叠条形图）
plt.figure(figsize=(8, 4))
x = np.arange(len(df))
win = df["OutcomeWin_pct"].values
stuck = df["OutcomeStuck_pct"].values
fail = df["OutcomeFail_pct"].values

plt.bar(x, win, label="Win (256)")
plt.bar(x, stuck, bottom=win, label="Stuck")
plt.bar(x, fail, bottom=win + stuck, label="Fail (early end)")
plt.xticks(x, df["Agent"], rotation=20)
plt.ylabel("Percentage (%)")
plt.title("Outcome Distribution (Unlimited steps)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------
# 3) 四宫格（2×2）
# ------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 左上：胜率
axes[0, 0].bar(df["Agent"], df["WinRate_pct"])
axes[0, 0].set_title("Win Rate (%)")
axes[0, 0].set_ylabel("Win Rate (%)")
axes[0, 0].tick_params(axis="x", rotation=20)

# 右上：AvgMaxTile（标注 256 比例）
axes[0, 1].bar(df["Agent"], df["AvgMaxTile"])
axes[0, 1].set_title("Avg Max Tile")
axes[0, 1].set_ylabel("Avg Max Tile")
axes[0, 1].tick_params(axis="x", rotation=20)
for i, r in df.iterrows():
    axes[0, 1].text(i, r["AvgMaxTile"], f'{r["Rate256_pct"]:.0f}% @256',
                    ha='center', va='bottom', fontsize=8)

# 左下：AvgTotalSteps
axes[1, 0].bar(df["Agent"], df["AvgTotalSteps"])
axes[1, 0].set_title("Avg Total Steps")
axes[1, 0].set_ylabel("Steps")
axes[1, 0].tick_params(axis="x", rotation=20)

# 右下：Outcome distribution（堆叠）
axes[1, 1].bar(x, win, label="Win")
axes[1, 1].bar(x, stuck, bottom=win, label="Stuck")
axes[1, 1].bar(x, fail, bottom=win + stuck, label="Fail")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(df["Agent"], rotation=20)
axes[1, 1].set_title("Outcome Distribution")
axes[1, 1].set_ylabel("Percent (%)")
axes[1, 1].legend()

plt.suptitle("2048 Agents Comparison (Unlimited steps)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
