import matplotlib.pyplot as plt
import numpy as np

# 数据
initial_weights = [20, 20, 20, 20, 15, 5]
adjusted_weights = [10, 25, 25, 20, 15, 5]
categories = ['Specific Time', 'Content Coverage', 'Language Diversity', 'Contextual Coherence', 'Grammar Accuracy', 'Spelling & Punctuation']

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 设置位置和宽度
x = np.arange(len(categories))
width = 0.35

# 绘制初始权重条形图
rects1 = ax.bar(x - width/2, initial_weights, width, label='Initial Weights')
# 绘制调整后权重条形图
rects2 = ax.bar(x + width/2, adjusted_weights, width, label='Adjusted Weights')

# 添加文本标签、标题和自定义 x 轴标签
ax.set_ylabel('Weights (%)')
ax.set_title('Comparison of Initial and Adjusted Weights by Criteria')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()

# 添加数据标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

# 调整布局
fig.tight_layout()

# 保存图形
plt.savefig("AHP_weight_changes_simple_final.png")

# 显示图形
plt.show()

