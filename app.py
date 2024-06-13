from flask import Flask, request, render_template
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 假设理想作文的文本和分数
ideal_essays = [
    "Your ideal essay 1 text...",
    "Your ideal essay 2 text...",
    "Your ideal essay 3 text...",
    "Your ideal essay 4 text...",
    "Your ideal essay 5 text...",
    "Your ideal essay 6 text...",
    "Your ideal essay 7 text...",
    "Your ideal essay 8 text...",
]

ideal_scores = [14, 14, 14, 14, 11, 8, 5, 2]

# 构建TF-IDF特征
vectorizer = TfidfVectorizer(stop_words='english')
X_ideal = vectorizer.fit_transform(ideal_essays)

# 使用LSA进行评分
lsa = TruncatedSVD(n_components=2)
X_lsa_ideal = lsa.fit_transform(X_ideal)

# 提取理想作文的向量
ideal_vectors = {score: X_lsa_ideal[i].reshape(1, -1) for i, score in enumerate(ideal_scores)}

# 使用MLR进行评分
model = LinearRegression()
model.fit(X_lsa_ideal, ideal_scores)


@app.route('/', methods=['GET', 'POST'])
def score_essay():
    if request.method == 'POST':
        essay = request.form['essay']

        # 将输入的作文转换为TF-IDF特征
        X_essay = vectorizer.transform([essay])
        X_lsa_essay = lsa.transform(X_essay)

        # 计算每篇作文与理想作文的余弦相似度
        similarity_scores = {score: cosine_similarity(X_lsa_essay, vec) for score, vec in ideal_vectors.items()}

        # 计算LSA评分
        high_score = similarity_scores[14][0][0]
        medium_score = similarity_scores[11][0][0]
        low_score = similarity_scores[8][0][0]
        very_low_score = similarity_scores[5][0][0]
        lowest_score = similarity_scores[2][0][0]
        final_score = high_score * 100 + medium_score * 75 + low_score * 50 + very_low_score * 25 + lowest_score * 10

        # 将LSA评分归一化到0到15的范围内
        lsa_scores = np.array([final_score])
        min_lsa_score, max_lsa_score = 0, 1400  # 假设最小和最大分数
        normalized_lsa_scores = 15 * (lsa_scores - min_lsa_score) / (max_lsa_score - min_lsa_score)

        # 使用MLR进行评分
        mlr_score = model.predict(X_lsa_essay)[0]

        # 计算最终分数
        final_score = 0.5 * normalized_lsa_scores[0] + 0.5 * mlr_score

        return render_template('score.html', score=final_score)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
