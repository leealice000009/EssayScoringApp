import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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

def score_essay(input_essay):
    # 文件路径
    essays_file_path = '43篇应用文.docx'
    ideal_essays_file_path = '8篇范文.docx'

    # 提取作文和理想作文
    documents = extract_text_from_docx(essays_file_path)
    ideal_essays = extract_text_from_docx(ideal_essays_file_path)

    # 理想作文的分数
    ideal_scores = [14, 14, 14, 14, 11, 8, 5, 2]

    # 合并所有文档（作文+理想作文）
    all_texts = documents + ideal_essays + [input_essay]

    # 构建TF-IDF特征
    vectorizer = TfidfVectorizer(stop_words='english')
    X_all = vectorizer.fit_transform(all_texts)

    # 分离TF-IDF特征
    X_documents = X_all[:len(documents)]
    X_ideal = X_all[len(documents):len(documents) + len(ideal_essays)]
    X_input = X_all[-1]

    # 使用LSA进行评分
    lsa = TruncatedSVD(n_components=2)
    X_lsa_all = lsa.fit_transform(X_all)

    # 分离LSA特征
    X_lsa_documents = X_lsa_all[:len(documents)]
    X_lsa_ideal = X_lsa_all[len(documents):len(documents) + len(ideal_essays)]
    X_lsa_input = X_lsa_all[-1].reshape(1, -1)

    # 提取理想作文的向量
    ideal_vectors = {score: X_lsa_ideal[i].reshape(1, -1) for i, score in enumerate(ideal_scores)}

    # 计算输入作文与理想作文的余弦相似度
    similarity_scores = {score: cosine_similarity(X_lsa_input, vec)[0][0] for score, vec in ideal_vectors.items()}

    # 计算输入作文的LSA评分
    high_score = similarity_scores[14]
    medium_score = similarity_scores[11]
    low_score = similarity_scores[8]
    very_low_score = similarity_scores[5]
    lowest_score = similarity_scores[2]
    final_score = high_score * 100 + medium_score * 75 + low_score * 50 + very_low_score * 25 + lowest_score * 10

    # 归一化LSA评分
    lsa_scores = np.array([final_score])
    min_lsa_score, max_lsa_score = lsa_scores.min(), lsa_scores.max()
    normalized_lsa_score = 15 * (lsa_scores - min_lsa_score) / (max_lsa_score - min_lsa_score)

    # 使用MLR进行评分
    model = LinearRegression()
    model.fit(X_lsa_ideal, ideal_scores)
    mlr_score = model.predict(X_lsa_input)

    # 计算最终分数
    final_score = 0.5 * normalized_lsa_score[0] + 0.5 * mlr_score[0]

    return final_score

# 测试函数
input_essay = "This is a test essay to be scored by the model."
score = score_essay(input_essay)
print(f"The score for the input essay is: {score:.4f}")
