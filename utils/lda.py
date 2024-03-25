
from pydoc import doc
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from sklearn.manifold import MDS
from matplotlib import pyplot as plt

def calculate_topic_similarity(topic1, topic2):
    return np.dot(topic1, topic2) / (np.linalg.norm(topic1) * np.linalg.norm(topic2))
def get_lda_message(posts):
    output = []
    stop_words = set(stopwords.words('english'))
    all_docs = posts['text_trans'].tolist()
    # 去掉url
    all_docs = [re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', doc) for doc in all_docs]
    tokenized = [doc.lower().split() for doc in all_docs]
    tokenized = [[word for word in doc if word not in stop_words] for doc in tokenized]
    if len(tokenized) == 0:
        output = []
    else:
        dictionary = corpora.Dictionary(tokenized)
        corpus = [dictionary.doc2bow(text) for text in tokenized]
        lda = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15, random_state=42)
        num_topics = lda.num_topics
        # 获取主题词名称
        topic_names = lda.print_topics(num_topics)
        processed_topic_names = []
        for topic in topic_names:
            # 使用正则表达式提取前三个词
            words = re.findall(r'"([^"]*)"', topic[1])[:3]
            # 使用下划线连接这些词，并添加到列表中
            processed_topic_names.append('_'.join(words))
        # print(processed_topic_names)
        topic_word_dist = lda.get_topics()
        document_topic_distributions = [lda.get_document_topics(doc) for doc in corpus]
        document_topics = [max(doc_topic, key=lambda x: x[1])[0] for doc_topic in document_topic_distributions]
        # print(f'doc_topics:{document_topics}')
        topic_size = []
        for i in range(num_topics):
            topic_size.append(document_topics.count(i + 1))
        topic_similarity_matrix = np.zeros((num_topics, num_topics))
        for i in range(num_topics):
            for j in range(num_topics):
                topic_similarity_matrix[i][j] = calculate_topic_similarity(topic_word_dist[i], topic_word_dist[j])
        # 将相似度矩阵转换为对称矩阵
        topic_similarity_matrix = 0.5 * (topic_similarity_matrix + topic_similarity_matrix.T)
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        topic_coordinates = mds.fit_transform(1 - topic_similarity_matrix)
        # 绘制intertopic距离图

        # 编码圆圈的大小以表示话题的数量
        max_size = max(topic_size)
        min_size = min(topic_size)
        for i in range(len(topic_size)):
            size = (topic_size[i] - min_size) / (max_size - min_size) * 100 + 10
            topic_size[i] = size
        for i in range(len(topic_size)):
            output.append({
                'x': topic_coordinates[i][0],
                'y': topic_coordinates[i][1],
                'interest_name': processed_topic_names[i],
                'size': topic_size[i]
            })

        # print(topic_size)
        # plt.figure(figsize=(10, 8))
        # for i in range(num_topics):
        #     plt.scatter(topic_coordinates[i, 0], topic_coordinates[i, 1], s=topic_size[i], alpha=0.5)
        # # 添加主题标签
        # for i in range(num_topics):
        #     plt.text(topic_coordinates[i, 0], topic_coordinates[i, 1], str(i), fontsize=12)
        # plt.xlabel('MDS Dimension 1')
        # plt.ylabel('MDS Dimension 2')
        # plt.title('Intertopic Distance Map')
        # plt.grid(True)
        # plt.show()
    return output