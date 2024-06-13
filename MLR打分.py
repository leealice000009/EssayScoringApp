import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取docx文件并提取每篇作文
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    texts = []
    current_essay = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("###"):  # 使用"###"作为每篇作文的分隔符
            if current_essay:
                texts.append("\n".join(current_essay))
                current_essay = []
        elif text:
            current_essay.append(text)
    if current_essay:
        texts.append("\n".join(current_essay))
    return texts

# 文件路径
essays_file_path = r"D:\Users\Alice\Documents\43篇应用文.docx"
ideal_essays_file_path = r"D:\Users\Alice\Documents\8篇范文.docx"

# 提取作文和理想作文
documents = extract_text_from_docx(essays_file_path)
ideal_essays = extract_text_from_docx(ideal_essays_file_path)

# 理想作文的分数
ideal_scores = [14, 14, 14, 14, 11, 8, 5, 2]

# 构建TF-IDF特征
vectorizer = TfidfVectorizer(stop_words='english')
X_ideal = vectorizer.fit_transform(ideal_essays)
X_documents = vectorizer.transform(documents)

# 训练多元线性回归模型
model = LinearRegression()
model.fit(X_ideal, ideal_scores)

# 预测43篇文章的分数
predicted_scores = model.predict(X_documents)

# 打印和可视化评分结果
final_scores = [(f"Essay{str(i+1).zfill(2)}", score) for i, score in enumerate(predicted_scores)]
final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
for essay, score in final_scores:
    print(f"{essay} Score: {score:.4f}")

# 可视化评分结果
essays, scores = zip(*final_scores)
plt.figure(figsize=(14, 12))
plt.barh(essays, scores, color='skyblue')
plt.xlabel('Scores')
plt.title('Essay Scores Based on MLR')
plt.gca().invert_yaxis()  # 翻转Y轴以便最高分在上
plt.show()
