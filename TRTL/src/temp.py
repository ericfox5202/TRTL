import matplotlib.pyplot as plt
import numpy as np

# 数据
layers = ['l=1', 'l=2', 'l=3']
avg_noSr = [11.8, 78.37, 1634]
avg_withSr = [11.82, 78.39, 1691.41]
max_noSr = [253, 754, 39541]
max_withSr = [253, 754, 39678]
mid_noSr = [3, 34, 173]
mid_withSr = [3, 34, 261]

x = np.arange(len(layers))
width = 0.1

# 绘制柱状图
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 2*width, avg_noSr, width, label='Avg (w/o $S_r$)')
ax.bar(x - width, avg_withSr, width, label='Avg (w/ $S_r$)')
ax.bar(x, max_noSr, width, label='Max (w/o $S_r$)')
ax.bar(x + width, max_withSr, width, label='Max (w/ $S_r$)')
ax.bar(x + 2*width, mid_noSr, width, label='Median (w/o $S_r$)')
ax.bar(x + 3*width, mid_withSr, width, label='Median (w/ $S_r$)')

# 样式设置
ax.set_xlabel('Expansion layer', fontsize=12)
ax.set_ylabel('Number of expanded edges (log scale)', fontsize=12)
ax.set_title('Edge expansion statistics per layer', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(layers, fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=10, frameon=False, ncol=2)
ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.6)

# 去除顶部和右侧边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()
