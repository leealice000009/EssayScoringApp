import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

# 合并所有文档
all_texts = documents + ideal_essays

# 构建TF-IDF特征
vectorizer = TfidfVectorizer(stop_words='english')
X_all = vectorizer.fit_transform(all_texts)

# 获取特征名称
feature_names = vectorizer.get_feature_names_out()
idf_scores = vectorizer.idf_

# 计算词频（TF）
tf_scores = np.sum(X_all.toarray(), axis=0)

# 显示词频和逆文档频率
tf_idf_data = list(zip(feature_names, tf_scores, idf_scores))
tf_idf_data = sorted(tf_idf_data, key=lambda x: x[1], reverse=True)  # 按照词频排序

# 提取前20个词的数据
top_n = 20
top_words = tf_idf_data[:top_n]
words, tfs, idfs = zip(*top_words)

# 可视化词频
plt.figure(figsize=(14, 7))
plt.barh(words, tfs, color='skyblue')
plt.xlabel('Term Frequency (TF)')
plt.title('Top 20 Words by Term Frequency')
plt.gca().invert_yaxis()  # 翻转Y轴以便最高频词在上
plt.show()

# 可视化逆文档频率
plt.figure(figsize=(14, 7))
plt.barh(words, idfs, color='lightgreen')
plt.xlabel('Inverse Document Frequency (IDF)')
plt.title('Top 20 Words by Inverse Document Frequency')
plt.gca().invert_yaxis()  # 翻转Y轴以便最高频词在上
plt.show()
