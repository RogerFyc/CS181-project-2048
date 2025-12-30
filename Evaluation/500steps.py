import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# 1) 录入你提供的数据（max steps=500）
# ------------------------
df = pd.DataFrame({
    "Agent": ["Minimax", "Actor-Critic", "Expectimax", "DQN"],
    "AvgScore": [316.8, 152.0, 115.2, 56.0],
    "SpecialOtherUtil_pct": [46.19, 86.00, 23.48, 75.63],  # special_other_util (%)
    "SpecialMaxUtil_pct": [0.00, 0.00, 1.14, 0.38],        # special_max_util (%)
    "CompletionRate_pct": [100, 35, 100, 85],              # 500-step completion (%)
})

# “到达较大数字”的摘要（你文字里给的 counts / 20 局）
df["Rate_512"]   = [6/20, 1/20, 0/20, 0/20]
df["Rate_ge256"] = [18/20, 5/20, 3/20, 0/20]

# 大数字得分：512 权重大一点（你可改）
df["BigTileScore"] = 2.0 * df["Rate_512"] + 1.0 * df["Rate_ge256"]

# 组合表现：AvgScore + BigTileScore 都归一化后加权
df["AvgScore_norm"] = df["AvgScore"] / df["AvgScore"].max()
df["BigTileScore_norm"] = df["BigTileScore"] / df["BigTileScore"].max() if df["BigTileScore"].max() > 0 else 0.0
df["Performance"] = 0.6 * df["AvgScore_norm"] + 0.4 * df["BigTileScore_norm"]

# ------------------------
# 2) 四张单独图
# ------------------------
plt.figure(figsize=(7, 4))
plt.bar(df["Agent"], df["AvgScore"])
plt.title("Average Score (Max 500 steps)")
plt.ylabel("Average Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(df["Agent"], df["SpecialOtherUtil_pct"])
plt.title("Special Tile Utilization (Other Tiles) - special_other_util")
plt.ylabel("Utilization (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(df["Agent"], df["SpecialMaxUtil_pct"])
plt.title("Special Tile Utilization (Max Tile) - special_max_util")
plt.ylabel("Utilization (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(df["Agent"], df["CompletionRate_pct"])
plt.title("Stability: 500-step Completion Rate")
plt.ylabel("Completion Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ------------------------
# 3) 四宫格（2×2）
# ------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].bar(df["Agent"], df["AvgScore"])
axes[0, 0].set_title("Average Score")
axes[0, 0].set_ylabel("Avg Score")
axes[0, 0].tick_params(axis="x", rotation=20)

axes[0, 1].bar(df["Agent"], df["SpecialOtherUtil_pct"])
axes[0, 1].set_title("Special Util (Other Tiles)")
axes[0, 1].set_ylabel("Util (%)")
axes[0, 1].tick_params(axis="x", rotation=20)

axes[1, 0].bar(df["Agent"], df["SpecialMaxUtil_pct"])
axes[1, 0].set_title("Special Util (Max Tile)")
axes[1, 0].set_ylabel("Util (%)")
axes[1, 0].tick_params(axis="x", rotation=20)

axes[1, 1].bar(df["Agent"], df["CompletionRate_pct"])
axes[1, 1].set_title("Completion Rate")
axes[1, 1].set_ylabel("Completion (%)")
axes[1, 1].tick_params(axis="x", rotation=20)

plt.suptitle("2048 Agents Comparison (Max 500 steps)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ------------------------
# 4) 拟合：Performance vs special_max_util
# ------------------------
x = df["SpecialMaxUtil_pct"].values
y = df["Performance"].values

# 线性拟合（注意：只有 4 个点，主要用于“展示趋势”，统计意义有限）
coef = np.polyfit(x, y, 1)
fit = np.poly1d(coef)

x_line = np.linspace(x.min(), x.max(), 100)
y_line = fit(x_line)

plt.figure(figsize=(7, 4))
plt.scatter(x, y)
for _, r in df.iterrows():
    plt.annotate(r["Agent"], (r["SpecialMaxUtil_pct"], r["Performance"]),
                 textcoords="offset points", xytext=(6, 6))
plt.plot(x_line, y_line)
plt.title("Performance vs special_max_util (Max-tile special usage)")
plt.xlabel("special_max_util (%)")
plt.ylabel("Performance (0.6*norm AvgScore + 0.4*norm BigTileReach)")
plt.tight_layout()
plt.show()

corr = np.corrcoef(x, y)[0, 1]
print(f"Linear fit: Performance = {coef[0]:.3f} * special_max_util + {coef[1]:.3f}")
print(f"Correlation: {corr:.3f}")
