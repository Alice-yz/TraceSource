
# 查找Top 10的关键词，全是英文
from collections import Counter
import re
def remove_stopwords(text, stopwords):
    words = text.split()  # 将文本分词为单词列表
    clean_words = [word for word in words if word not in stopwords]  # 去除停用词
    clean_text = ' '.join(clean_words)  # 将列表中的单词重新组合成文本
    return clean_text
def remove_newlines(text):
    # 将换行符替换为空格
    clean_text = text.replace('\n', ' ')
    return clean_text
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # 将匹配到的网址替换为空字符串
    clean_text = url_pattern.sub('', text)
    return clean_text
def remove_after_at(text):
    # 匹配@符号后面的单词的正则表达式
    after_at_pattern = re.compile(r'@\w+\s?')
    clean_text = after_at_pattern.sub('', text)
    return clean_text
def remove_punctuation(text):
    clean_text = re.sub(r'[^\w\s]', '', text)
    return clean_text
def convert_to_lowercase(text):
    return text.lower()
english_stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now' ,'RT' ,'weibo', 'rt' ,'cctv' ,'time',
    'daiichi' ,'the' ,'fukushima' ,'[#' ,'#]' ,'5th'
]
def preprocess_text(text, stopwords):
    text = remove_newlines(text)
    text = remove_urls(text)
    text = remove_after_at(text)
    text = remove_punctuation(text)
    text = convert_to_lowercase(text)
    text = remove_stopwords(text, stopwords)
    return text
def find_top_n_words(text, n):
    words = ' '.join(text).split()  # 将所有文本拼接成一个长字符串后分词
    word_freq = Counter(words)  # 统计词频
    top_n_words = word_freq.most_common(n)  # 获取词频最高的前n个单词
    return top_n_words


def get_word_freq_list(text):
    text = text.apply(lambda x: preprocess_text(x, english_stopwords))
    top_words = find_top_n_words(text, 10)
    return top_words
