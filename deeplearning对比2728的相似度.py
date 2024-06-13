import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import docx

# 读取文档内容
def read_docx(file_path):
    doc = docx.Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

# 加载文档内容
text1 = read_docx('C:/Users/Alice/Documents/作文27原文.docx')
text2 = read_docx('C:/Users/Alice/Documents/作文28原文.docx')

# 加载预训练的 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    # 对文本进行编码
    inputs = tokenizer(text, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
    # 使用 BERT 模型进行嵌入
    outputs = model(inputs)
    # 获取最后一层的隐藏状态
    embeddings = outputs.last_hidden_state
    # 取平均值作为句子嵌入
    return tf.reduce_mean(embeddings, axis=1)

def cosine_similarity(embedding1, embedding2):
    # 计算余弦相似度
    dot_product = tf.reduce_sum(embedding1 * embedding2, axis=1)
    norm1 = tf.norm(embedding1, axis=1)
    norm2 = tf.norm(embedding2, axis=1)
    return dot_product / (norm1 * norm2)

# 获取文本嵌入
embedding1 = embed_text(text1)
embedding2 = embed_text(text2)

# 计算相似度
similarity = cosine_similarity(embedding1, embedding2)

print(f"Similarity: {similarity.numpy()[0]:.4f}")
