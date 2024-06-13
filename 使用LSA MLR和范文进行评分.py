import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
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

# 合并所有文档（作文+理想作文）
all_texts = documents + ideal_essays

# 构建TF-IDF特征
vectorizer = TfidfVectorizer(stop_words='english')
X_all = vectorizer.fit_transform(all_texts)

# 分离TF-IDF特征
X_documents = X_all[:len(documents)]
X_ideal = X_all[len(documents):]

# 使用LSA进行评分
lsa = TruncatedSVD(n_components=2)
X_lsa_all = lsa.fit_transform(X_all)

# 分离LSA特征
X_lsa_documents = X_lsa_all[:len(documents)]
X_lsa_ideal = X_lsa_all[len(documents):]

# 提取理想作文的向量
ideal_vectors = {score: X_lsa_ideal[i].reshape(1, -1) for i, score in enumerate(ideal_scores)}

# 计算每篇作文与理想作文的余弦相似度
similarity_scores = {score: cosine_similarity(X_lsa_documents, vec) for score, vec in ideal_vectors.items()}

# 打印每篇作文与不同理想分数范文的相似度结果
for i in range(len(documents)):
    print(f"Essay{i+1:02d}:")
    for score in ideal_scores:
        print(f"  Similarity to {score}-point ideal essay: {similarity_scores[score][i][0]:.4f}")
    print()

# 计算每篇作文的LSA评分
lsa_scores = []
for i in range(len(documents)):
    high_score = similarity_scores[14][i][0]
    medium_score = similarity_scores[11][i][0]
    low_score = similarity_scores[8][i][0]
    very_low_score = similarity_scores[5][i][0]
    lowest_score = similarity_scores[2][i][0]
    final_score = high_score * 100 + medium_score * 75 + low_score * 50 + very_low_score * 25 + lowest_score * 10
    lsa_scores.append(final_score)

# 将LSA评分归一化到0到15的范围内
lsa_scores = np.array(lsa_scores)
min_lsa_score, max_lsa_score = lsa_scores.min(), lsa_scores.max()
normalized_lsa_scores = 15 * (lsa_scores - min_lsa_score) / (max_lsa_score - min_lsa_score)

# 使用MLR进行评分
model = LinearRegression()
model.fit(X_lsa_ideal, ideal_scores)
mlr_scores = model.predict(X_lsa_documents)

# 计算最终分数
final_scores = 0.5 * normalized_lsa_scores + 0.5 * mlr_scores  # 结合LSA和MLR的评分

# 生成最终评分表
final_score_list = [(f"Essay{str(i+1).zfill(2)}", score) for i, score in enumerate(final_scores)]

# 按顺序输出分数
for essay, score in final_score_list:
    print(f"{essay} Score: {score:.4f}")

# 按顺序可视化最终评分结果
essays, scores = zip(*final_score_list)
plt.figure(figsize=(14, 12))
plt.barh(essays, scores, color='skyblue')
plt.xlabel('Scores')
plt.title('Final Essay Scores Based on Combined LSA and MLR')
plt.gca().invert_yaxis()  # 翻转Y轴以便最高分在上
plt.show()
