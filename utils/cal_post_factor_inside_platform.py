from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
from pydoc import doc
import gensim
from gensim import corpora
from pprint import pprint
import re
import pandas as pd
import numpy as np
from utils import cal_cluster_factor as ccf
from langdetect import detect
import nltk
from nltk.corpus import stopwords
# 下载停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model_eng = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model_mut = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
def calculate_bert_similarity(text1, text2,model):
    embeddings = model.encode([text1, text2])
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return sim.tolist()[0][0]
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer()
    corpus = [text1, text2]
    vectors = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1]
def find_url_from_text(text):
    return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

def check_source_platform(B_text: object, A_platform: object) -> object:
    special_words = []
    if A_platform == 'weibo':
        special_list = ['weibo','Weibo','Chinese Twitter','chinese twitter']
    elif A_platform == 'twitter':
        special_list = ['twitter','Twitter','bluebird','Bluebird']
    else:
        special_list = ['facebook','Facebook']
    for special in special_list:
        if special in B_text:
            special_words.append(special)
    return special_words

def check_special_info(text,src_platform,src_usernames):
    # 返回分数以及包含的特定的词
    special_info = ["source", "cr.", "original post", "source:", "via", "via:", "original link", "credit:", "from",
                    "from:","original link","courtesy of","reprinted from","quote form","cited from","hat tip","originally published by"]
    special_words = check_source_platform(text,src_platform)

    score = 0
    for info in special_info:
        if info in text.lower():
            score = 0.5
            special_words.append(info)
    for src_username in src_usernames:
        if src_username is None:
            continue
        if src_username.lower() in text.lower():
            score = 1
            special_words.append(src_username.lower())
    return score,special_words


def find_hashtags(text):
    return re.findall(r'#\w+#|\B#\w+\b', text)


def cal_post_factor(post_A,post_B,df_data,self,debug=False):
    # 首先判断谁传给谁
    if post_A['publish_time'] > post_B['publish_time']:
        s_post = post_B
        t_post = post_A
    else:
        s_post = post_A
        t_post = post_B
    s_platform = s_post['from']
    t_platform = t_post['from']
    ########### Recent Time ###########
    s_date = pd.to_datetime(s_post['publish_time'])
    t_date = pd.to_datetime(t_post['publish_time'])
    time_diff = (t_date - s_date).days
    # print(f"time_diff: {time_diff}")
    ########### Sim words ###########
    word_sim = calculate_bert_similarity(s_post['text_trans'], t_post['text_trans'], model_eng)
    # print(f"sim: {word_sim}")
    ########### Sim URL ###########
    s_url = set(find_url_from_text(s_post['text_trans']))
    t_url = set(find_url_from_text(t_post['text_trans']))
    url_intersection = s_url.intersection(t_url)
    url_union = s_url.union(t_url)
    url_sim = len(url_intersection) / len(url_union) if len(url_union) != 0 else 0
    # print(f"s_url_index: {s_url_index}, t_url_index: {t_url_index}")
    # print(f"url_sim: {url_sim}")
    ########### DirectURL #############
    s_post_url = s_post['url']
    if s_post_url in t_url:
        direct_url = 1
    else:
        direct_url = 0
    ########### Sim Hashtag ###########
    s_hashtag = find_hashtags(s_post['text_trans'])
    t_hashtag = find_hashtags(t_post['text_trans'])
    if len(s_hashtag) != 0 and len(t_hashtag) != 0:
        hashtag_sim = calculate_bert_similarity(' '.join(s_hashtag), ' '.join(t_hashtag), model_eng)
    else:
        hashtag_sim = 0
    same_hashtag = list(set(s_hashtag).intersection(set(t_hashtag)))
    for hashtag in same_hashtag:
        s_hashtag_begin_idx = s_post['text_trans'].find(hashtag)
        t_hashtag_begin_idx = t_post['text_trans'].find(hashtag)
        s_hashtag_end_idx = s_hashtag_begin_idx + len(hashtag)
        t_hashtag_end_idx = t_hashtag_begin_idx + len(hashtag)
    # print(f"hashtag_sim: {hashtag_sim}")
    # ########### Inf post ###########
    s_platform_posts = df_data[df_data['from'] == s_platform]
    t_platform_posts = df_data[df_data['from'] == t_platform]
    # 计算当前最高的inf_posts
    max_engagement_s = s_platform_posts[["cnt_retweet", "cnt_agree", "cnt_comment"]].sum(axis=1).idxmax()
    max_engagement_t = t_platform_posts[["cnt_retweet", "cnt_agree", "cnt_comment"]].sum(axis=1).idxmax()
    # 计算s、t的inf_posts
    engagement_t = t_post["cnt_retweet"] + t_post["cnt_agree"] + t_post["cnt_comment"]
    engagement_s = s_post["cnt_retweet"] + s_post["cnt_agree"] + s_post["cnt_comment"]
    # print(f"max_s: {max_engagement_s},max_t: {max_engagement_t},engagement_s: {engagement_s}, engagement_t: {engagement_t}")
    penalty = 0.2
    if t_platform == 'weibo':
        penalty = 0.5
    is_post = not s_post['action'] == 'post'
    inf = np.log(1 + engagement_s) / np.log(1 + max_engagement_s) * (1 - penalty * is_post)
    # print(f"inf: {inf}")
    # 检查t平台的特征关键词
    special_score,special_words = check_special_info(t_post['text_trans'], s_post['from'], [self.get_user_info(s_post['user_id'])['name'], self.get_user_info(s_post['user_id'])['screen_name_trans']])
    for word in special_words:
        s_special_begin_idx = s_post['text_trans'].find(word)
        t_special_begin_idx = t_post['text_trans'].find(word)
        s_special_end_idx = s_special_begin_idx + len(word)
        t_special_end_idx = t_special_begin_idx + len(word)
    # print(f"special_score: {special_score}")
    ########### Sim User info ###########
    s_user = self.get_user_info(s_post['user_id'])
    t_user = self.get_user_info(t_post['user_id'])
    # 不跨语言用jarosim
    jaro_sim = Levenshtein.jaro(s_user['name'], t_user['name'])
    # 跨语言用bert
    bert_sim = calculate_bert_similarity(s_user['name'], t_user['name'], model_mut)
    if detect(s_user['name']) != detect(t_user['name']):
        username_sim = bert_sim
    else:
        username_sim = jaro_sim
    # print(f"jaro_sim: {jaro_sim}")
    ########### Sim User description ###########
    s_description = s_user['description']
    t_description = t_user['description']
    if type(s_description) != str:
        if pd.isna(s_description):
            s_description = s_user['name']
        else:
            s_description = s_description.values[0]
    if type(t_description) != str:
        if pd.isna(t_description):
            t_description = t_user['name']
        else:
            t_description = t_description.values[0]
    sim_description = calculate_bert_similarity(s_description, t_description, model_mut)
    # print(f"sim_description: {sim_description}")
    sim_userinfo = 0.5 * username_sim + 0.5 * sim_description
    if t_user['type'] ==None:
        t_SNS = -1
    else:
        t_SNS = 1
    word_list_1 = ['america', 'USA','U.S.', 'twitter', 'facebook']
    word_list_2 = ['china','chinese', 'weibo']
    flag = False
    if t_post['from'] == 'weibo':
        for word in word_list_1:
            if word in t_post['text_trans']:
                flag = True
                break
    else:
        for word in word_list_2:
            if word in t_post['text_trans']:
                flag = True
                break
    if sim_userinfo > 0.5:
        diffusion_pattern_type = 0
    elif 0.3 < sim_userinfo <= 0.5 or t_SNS > 0 or flag:
        diffusion_pattern_type = 1
    else:
        diffusion_pattern_type = 2
    ########### Sim User name ###########
    # 寻找单个账号粉丝量最大的
    max_fans = 0
    for idx, row in s_platform_posts.iterrows():
        user = self.get_user_info(row['user_id'])
        if int(user['fan']) > max_fans:
            max_fans = int(user['fan'])
    norm_fans = int(s_user['fan']) / max_fans
    is_verified = 1 if s_user['validation'] == 'True' else 0
    inf_user = 0.6 * norm_fans + 0.2 * is_verified
    # print(f"inf_user: {inf_user}")
    ########### Sim Interest ###########
    s_all_posts = df_data[(df_data['user_id'] == s_post['user_id']) & (df_data['post_id'] != s_post['post_id'])]
    t_all_posts = df_data[(df_data['user_id'] == t_post['user_id']) & (df_data['post_id'] != t_post['post_id'])]
    # print(f"s_all_posts: {s_all_posts.shape[0]}, t_all_posts: {t_all_posts.shape[0]}")
    s_all_docs = s_all_posts['text_trans'].tolist()
    t_all_docs = t_all_posts['text_trans'].tolist()
    s_tokenized = [doc.lower().split() for doc in s_all_docs]
    t_tokenized = [doc.lower().split() for doc in t_all_docs]
    s_tokenized = [[word for word in doc if word not in stop_words] for doc in s_tokenized]
    t_tokenized = [[word for word in doc if word not in stop_words] for doc in t_tokenized]
    if s_tokenized == [] or t_tokenized == []:
        sim_interest = 0
    else:
        s_dictionary = corpora.Dictionary(s_tokenized)
        t_dictionary = corpora.Dictionary(t_tokenized)
        s_corpus = [s_dictionary.doc2bow(text) for text in s_tokenized]
        t_corpus = [t_dictionary.doc2bow(text) for text in t_tokenized]
        s_lda = gensim.models.LdaModel(s_corpus, num_topics=10, id2word=s_dictionary, passes=15, random_state=42)
        t_lda = gensim.models.LdaModel(t_corpus, num_topics=10, id2word=t_dictionary, passes=15, random_state=42)
        # pprint(s_lda.print_topics())
        # pprint(t_lda.print_topics())
        s_dist = s_lda.get_document_topics(s_corpus)
        t_dist = t_lda.get_document_topics(t_corpus)
        kl_divergence = 0
        for s_doc, t_doc in zip(s_dist, t_dist):
            for (topic, prob) in s_doc:
                t_prob = 0
                # print(f"topic: {topic}, prob: {prob}")
                for (t_topic, t_prob) in t_doc:
                    if t_topic == topic:
                        t_prob = t_prob
                        break
                prob = round(prob, 2)
                t_prob = round(t_prob, 2)
                kl_divergence += prob * np.log(prob / t_prob)
        # print(f"kl_divergence: {kl_divergence}")
        sim_interest = 1 / (1 + kl_divergence)
    # print(f"sim_interest: {sim_interest}")
    ########### PastSpread  ###########
    s_user_all_post_url = ccf.find_posts_with_url(s_all_posts, 'text')
    t_user_all_post_url = ccf.find_posts_with_url(t_all_posts, 'text')
    s_user_all_post_url_list, s_user_all_post_id_list = ccf.find_same_url(s_user_all_post_url, 'text')
    t_user_all_post_url_list, t_user_all_post_id_list = ccf.find_same_url(t_user_all_post_url, 'text')
    same_url = list(set(s_user_all_post_url_list).intersection(set(t_user_all_post_url_list)))
    same_url_count = len(same_url)
    ########### Sim Past hashtag ###########
    s_user_all_post_hashtag = ccf.find_hashtags(s_all_posts)
    t_user_all_post_hashtag = ccf.find_hashtags(t_all_posts)
    s_user_all_post_hashtag = [item for sublist in s_user_all_post_hashtag for item in sublist]
    t_user_all_post_hashtag = [item for sublist in t_user_all_post_hashtag for item in sublist]
    s_user_all_post_hashtag = list(set(s_user_all_post_hashtag))
    t_user_all_post_hashtag = list(set(t_user_all_post_hashtag))
    same_hashtag = list(set(s_user_all_post_hashtag).intersection(set(t_user_all_post_hashtag)))
    same_hashtag_count = len(same_hashtag)
    ######### DirectPastURL #########
    s_all_url = s_all_posts['url'].dropna().tolist()
    s_all_url = list(set(s_all_url))
    same_url = list(set(s_all_url).intersection(set(t_user_all_post_url_list)))
    direct_past_url = 1 if len(same_url) > 0 else 0
    past_spread = 0.5 * direct_past_url + 0.25 * same_url_count + 0.25 * same_hashtag_count
    # print(f"direc_url_count: {direct_url_count}, same_url_count: {same_url_count}, same_hashtag_count: {same_hashtag_count}")
    # print(f"past_spread: {past_spread}")
    ########### NumPath ###########
    # 求出t针对s的topic
    s_topic = s_post['cluster']
    t_topic = t_post['cluster']
    t_topic_post = t_all_posts[t_all_posts['cluster'] == t_topic]
    s_topic_post = s_all_posts[s_all_posts['cluster'] == s_topic]
    num_path = t_topic_post.shape[0] + s_topic_post.shape[0]
    # print(f"num_path: {num_path}")
    post_rel = max(direct_url, special_score, max(word_sim, hashtag_sim, url_sim))
    usr_rel = inf_user # * 0.5 * (username_sim + sim_description)
    his_rel = 2 * (0.4 * sim_interest + 0.6 * past_spread) / (1 + np.exp(-num_path))
    # print(f"post_rel: {post_rel}, usr_rel: {usr_rel}, his_rel: {his_rel}")
    factor = 1 - np.exp(-(post_rel + usr_rel + his_rel) / (1 + 0.2 * time_diff))
    factor = max(direct_url, factor)
    # print(f"factor: {factor}")
    # 把上面所有上面的print
    if debug:
        print(f"time_diff: {time_diff}")
        print(f"sim: {word_sim}")
        print(f"url_sim: {url_sim}")
        print(f"hashtag_sim: {hashtag_sim}")
        print(f"inf: {inf}")
        print(f"special_score: {special_score}")
        print(f"jaro_sim: {username_sim}")
        print(f"sim_description: {sim_description}")
        print(f"inf_user: {inf_user}")
        print(f"sim_interest: {sim_interest}")
        print(f"past_spread: {past_spread}")
        print(f"num_path: {num_path}")
        print(f"post_rel: {post_rel}, usr_rel: {usr_rel}, his_rel: {his_rel}")
        print(f"factor: {factor}")
    return factor,post_rel,special_score,s_post,t_post
