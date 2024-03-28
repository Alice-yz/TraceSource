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
import json

# 下载停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model_eng = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model_mut = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


def calculate_bert_similarity(text1, text2, model):
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


def check_special_info(text):
    # 返回分数以及包含的特定的词
    special_info = ["source", "cr.", "original post", "source:", "via", "via:", "original link", "credit:", "from",
                    "from:", "original link", "courtesy of", "reprinted from", "quote form", "cited from", "hat tip",
                    "originally published by"]
    special_mention = ["twitter", "bluebird", "facebook", "weibo", "chinese twitter", "Chinese Twitter", "weibo.com"]
    special_words = []
    score = 0
    for info in special_info:
        if info in text:
            score = 0.5
            special_words.append(info)
    for mention in special_mention:
        if mention in text:
            score = 1
            special_words.append(mention)
    return score, special_words


def find_hashtags(text):
    return re.findall(r'#\w+#|\B#\w+\b', text)


def get_user_info(all_users, user_id):
    user_info = all_users[all_users['user_id'] == user_id]
    if user_info.empty:
        # 最后一行是假数据
        user_info = all_users.iloc[-1]
    else:
        user_info = user_info.iloc[0]
    return user_info


def cal_post_factor(post_A, post_B, df_data, self, debug=False):
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
    # 查找same_url在s_post和t_post中的位置
    same_url = list(url_intersection)
    s_highlight = []
    t_highlight = []
    for url in same_url:
        s_url_begin_idx = s_post['text_trans'].find(url)
        t_url_begin_idx = t_post['text_trans'].find(url)
        s_url_end_idx = s_url_begin_idx + len(url)
        t_url_end_idx = t_url_begin_idx + len(url)
        s_highlight.append({
            "begin": int(s_url_begin_idx),
            "end": int(s_url_end_idx)
        })
        t_highlight.append({
            "begin": int(t_url_begin_idx),
            "end": int(t_url_end_idx)
        })
    # print(f"s_url_index: {s_url_index}, t_url_index: {t_url_index}")
    # print(f"url_sim: {url_sim}")
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
        s_highlight.append({
            "begin": int(s_hashtag_begin_idx),
            "end": int(s_hashtag_end_idx)
        })
        t_highlight.append({
            "begin": int(t_hashtag_begin_idx),
            "end": int(t_hashtag_end_idx)
        })
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
    special_score, special_words = check_special_info(t_post['text_trans'])
    for word in special_words:
        s_special_begin_idx = s_post['text_trans'].find(word)
        t_special_begin_idx = t_post['text_trans'].find(word)
        s_special_end_idx = s_special_begin_idx + len(word)
        t_special_end_idx = t_special_begin_idx + len(word)
        s_highlight.append({
            "begin": s_special_begin_idx,
            "end": s_special_end_idx
        })
        t_highlight.append({
            "begin": t_special_begin_idx,
            "end": t_special_end_idx
        })
    # print(f"special_score: {special_score}")
    ########### Sim User info ###########
    s_user = get_user_info(df_all_accounts, s_post['user_id'])
    t_user = get_user_info(df_all_accounts, t_post['user_id'])
    # 不跨语言用jarosim
    jaro_sim = Levenshtein.jaro(s_user['name'], t_user['name'])
    # 跨语言用bert
    bert_sim = calculate_bert_similarity(s_user['name'], t_user['name'], model_mut)
    if s_user['name'] == None or pd.isna(s_user['name']) or t_user['name'] == None or pd.isna(t_user['name']):
        username_sim = 0
    else:
        try:
            if detect(s_user['name']) != detect(t_user['name']):
                username_sim = bert_sim
            else:
                username_sim = jaro_sim
        except:
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
    if t_user['type'] == None or pd.isna(t_user['type']):
        t_SNS = -1
    else:
        t_SNS = 1
    word_list_1 = ['america', 'USA', 'U.S.', 'twitter', 'facebook']
    word_list_2 = ['china', 'chinese', 'weibo']
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
    if sim_userinfo >= 0.5:
        diffusion_pattern_type = 0
    elif 0.3 <= sim_userinfo < 0.5 or t_SNS > 0 or flag:
        diffusion_pattern_type = 1
    else:
        diffusion_pattern_type = 2
    ########### Sim User name ###########
    # 寻找单个账号粉丝量最大的
    max_fans = 0
    for idx, row in s_platform_posts.iterrows():
        user = get_user_info(df_all_accounts, row['user_id'])
        fan_count = int(user['fan'])
        if fan_count > max_fans:
            max_fans = int(user['fan'])
    norm_fans = int(s_user['fan']) / max_fans
    is_verified = 1 if s_user['validation'] == 'True' else 0
    inf_user = 0.6 * norm_fans + 0.2 * is_verified
    # print(f"inf_user: {inf_user}")
    ########### Sim Interest ###########
    s_all_posts = df_data[df_data['user_id'] == s_post['user_id']]
    t_all_posts = df_data[df_data['user_id'] == t_post['user_id']]
    # print(f"s_all_posts: {s_all_posts.shape[0]}, t_all_posts: {t_all_posts.shape[0]}")
    s_all_docs = s_all_posts['text_trans'].tolist()
    t_all_docs = t_all_posts['text_trans'].tolist()
    s_tokenized = [doc.lower().split() for doc in s_all_docs]
    t_tokenized = [doc.lower().split() for doc in t_all_docs]
    s_tokenized = [[word for word in doc if word not in stop_words] for doc in s_tokenized]
    t_tokenized = [[word for word in doc if word not in stop_words] for doc in t_tokenized]
    try:
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
    except:
        sim_interest = 0
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
    ######### DirectURL #########
    s_all_url = s_all_posts['url'].dropna().tolist()
    t_all_url = t_all_posts['url'].dropna().tolist()
    s_all_url = list(set(s_all_url))
    t_all_url = list(set(t_all_url))
    same_url = list(set(s_all_url).intersection(set(t_all_url)))
    direct_url_count = len(same_url)
    past_spread = 0.5 * direct_url_count + 0.25 * same_url_count + 0.25 * same_hashtag_count
    # print(f"direc_url_count: {direct_url_count}, same_url_count: {same_url_count}, same_hashtag_count: {same_hashtag_count}")
    # print(f"past_spread: {past_spread}")
    ########### NumPath ###########
    # 求出t针对s的topic
    s_topic = s_post['cluster']
    t_topic_post = t_all_posts[t_all_posts['cluster'] == s_topic]
    t_topic = t_post['cluster']
    s_topic_post = s_all_posts[s_all_posts['cluster'] == t_topic]
    num_path = t_topic_post.shape[0] + s_topic_post.shape[0]
    # print(f"num_path: {num_path}")
    post_rel = 0.5 * max(2 * direct_url_count, special_score) + inf * max(word_sim, hashtag_sim, url_sim)
    usr_rel = inf_user * 0.5 * (username_sim + sim_description)
    his_rel = 2 * (0.4 * sim_interest + 0.6 * past_spread) / (1 + np.exp(-num_path))
    # print(f"post_rel: {post_rel}, usr_rel: {usr_rel}, his_rel: {his_rel}")
    factor = 1 - np.exp(-(post_rel + usr_rel + his_rel) / (1 + 0.2 * time_diff))
    factor = max(direct_url_count, factor)
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
    return factor, s_post, t_post, s_highlight, t_highlight, diffusion_pattern_type


def format_post(s_post, t_post, s_highlight, t_highlight, factor, diffusion_pattern,is_assigned):
    return {
        "source":{
            "id": s_post["post_id"],
            "platform": s_post["from"],
            'highlight': s_highlight,
            'text': str(s_post['text'])
        }
        ,
        "target": {
            "id": t_post["post_id"],
            "platform": t_post["from"],
            "highlight": t_highlight,
            'text': str(t_post['text'])
        },
        "width": factor,
        "diffusion_pattern": diffusion_pattern,
        "is_assigned": is_assigned
    }


if __name__ == '__main__':
    df_all_posts = pd.read_csv('all_posts.csv', dtype={'user_id': str, 'post_id': str})
    df_all_posts.where(df_all_posts.notnull(), None)
    df_all_accounts = pd.read_csv('all_accounts.csv', dtype={'user_id': str})
    df_all_accounts.where(df_all_accounts.notnull(), None)
    output = {}
    hash_table = [
        ('Great_Wave_Kanagawa', "2021-04-20", "2021-04-29"),
        ('foreign_affairs_questions', '2021-04-20', '2021-04-29'),
        ('japan_nuclear_wastewater', '2021-04-20', '2021-04-29'),
        ('radioactive_condemn_water', '2021-04-20', '2021-04-29'),
        ('240_china_nuclear_pollution', '2023-08-21', '2023-08-30'),
        ('70_billion_japan_water', '2023-08-21', '2023-08-30'),
        ('cooling_water_nuclear_wastewater', '2023-08-21', '2023-08-30'),
        ('south_korea_nuclear_discharge', '2023-08-21', '2023-09-01'),
        ('sue_TEPCO_japan', '2023-08-21', '2023-08-30'),
        ('radioactive_pollution_japan_sea', '2023-08-21', '2023-08-30'),
        ('treatment_japan_waste_nuclear', '2023-08-21', '2023-08-30'),
        ('japan_dead_fish', '2023-12-01', '2023-12-10'),
    ]
    cluster_names = [item[0] for item in hash_table]
    # 提取中间一列作为 Python 列表
    # debug = True
    for idx in range(len(hash_table)):
        if idx == 1 or idx == 2 or idx == 3 or idx == 9 or idx == 10 or idx == 11:
            continue
        # idx = 11
        output_cluster = []
        start_time = hash_table[idx][1]
        end_time = hash_table[idx][2]
        cluster = cluster_names[idx]
        # end_time 加一天
        end_time = (pd.to_datetime(end_time) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        A = 'weibo'
        B = 'twitter'
        df_data = df_all_posts[
            (df_all_posts['publish_time'] >= start_time) & (df_all_posts['publish_time'] <= end_time)]
        # 要求query不是highlight
        df_data = df_data[df_data['query'] != 'highlight']
        # df_data = df_data[df_data['query'] == 'highlight']
        df_data = df_data[df_data['cluster'] == cluster]
        df_data_A = df_data[df_data['from'] == A]
        df_data_B = df_data[df_data['from'] == B]
        print(f"Cluster: {cluster}  df_data shape: {df_data.shape[0]} df_data_A shape: {df_data_A.shape[0]} df_data_B shape: {df_data_B.shape[0]}")
        debug = False
        post_num = 0
        ### 先解决PPT中提到的
        select_data = []
        idx_rand_a = 0
        idx_rand_b = 0
        select_data.append((idx_rand_a, idx_rand_b))
        for idx_a in range(df_data_A.shape[0]):
            for idx_b in range(df_data_B.shape[0]):
                if (idx_rand_a, idx_rand_b) in select_data:
                    idx_rand_a = np.random.randint(0, df_data_A.shape[0])
                    idx_rand_b = np.random.randint(0, df_data_B.shape[0])
                    while (idx_rand_a, idx_rand_b) in select_data:
                        idx_rand_a = np.random.randint(0, df_data_A.shape[0])
                        idx_rand_b = np.random.randint(0, df_data_B.shape[0])
                    select_data.append((idx_rand_a, idx_rand_b))
                post_num += 1
                factor, s_post, t_post, s_highlight, t_highlight, diffusion_pattern = cal_post_factor(
                    df_data_A.iloc[idx_rand_a], df_data_B.iloc[idx_rand_b], df_data, debug, debug)
                print(f"{post_num}:Cluster: {cluster}({idx_rand_a},{idx_rand_b})-|- {s_post['from']}-->{t_post['from']} ,Factor: {factor} Diffusion Pattern Type: {diffusion_pattern}")
                # print(f"source:{s_post['text']}")
                # print(f"target:{t_post['text']}")
                # input_str =  input("Enter y is assigned, n is not assigned:\n")
                # is_assigned = True if input_str == 'y' else False
                output_cluster.append(format_post(s_post, t_post, s_highlight, t_highlight, factor, diffusion_pattern,False))
                # print(is_assigned)

                if post_num > 50:
                    break
            if post_num > 50:
                break
        output[cluster] = output_cluster
    # 写入文件
    with open(f'./json/post_other.json', 'w',encoding='utf-8') as f:
        f.write(json.dumps(output, ensure_ascii=False,indent=4))
