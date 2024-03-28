######
from sentence_transformers import SentenceTransformer, util
import os
from utils import freq
import re
import pandas as pd
import numpy as np

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def cal_RW_SCORE(A_posts, B_posts):
    A_texts = freq.get_word_freq_list(A_posts['text_trans'])
    B_texts = freq.get_word_freq_list(B_posts['text_trans'])
    # print(A_texts, B_texts)
    A_word_list = [word[0] for word in A_texts]
    B_word_list = [word[0] for word in B_texts]
    embed_A = model.encode(A_word_list, convert_to_tensor=True)
    embed_B = model.encode(B_word_list, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embed_A, embed_B)
    cosine_scores = cosine_scores.cpu().numpy()
    rw_res = []
    for i in range(len(A_word_list)):
        # print(f"{A_word_list[i]} <--> {B_word_list[cosine_scores[i].argmax()]}, score: {cosine_scores[i].max()}")
        rw_res.append({
            'source': A_word_list[i],
            'target': B_word_list[cosine_scores[i].argmax()],
            'score': cosine_scores[i].max()
        })
    if len(rw_res) == 0:
        rw_score = 0
    else:
        rw_score = sum([item['score'] for item in rw_res]) / len(rw_res)
    # print(f"rw_score: {rw_score}")
    return rw_score, rw_res


def cal_hashtag_score(A_posts, B_posts):
    A_hashtags = find_hashtags(A_posts).tolist()
    B_hashtags = find_hashtags(B_posts).tolist()
    A_hashtags_list = [item for sublist in A_hashtags for item in sublist]
    B_hashtags_list = [item for sublist in B_hashtags for item in sublist]
    if len(A_hashtags_list) == 0 or len(B_hashtags_list) == 0:
        hashtag_score = 0
        hashtag_res = []
        return hashtag_score, hashtag_res
    # print(f"A_hashtags: {A_hashtags_list}")
    # print(f"B_hashtags: {B_hashtags_list}")
    embed_A = model.encode(A_hashtags_list, convert_to_tensor=True)
    embed_B = model.encode(B_hashtags_list, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embed_A, embed_B)
    cosine_scores = cosine_scores.cpu().numpy()
    hashtag_res = []
    for i in range(len(A_hashtags_list)):
        hashtag_res.append({
            'source': A_hashtags_list[i],
            'target': B_hashtags_list[cosine_scores[i].argmax()],
            'score': cosine_scores[i].max()
        })
    if len(hashtag_res) == 0:
        hashtag_score = 0
    else:
        hashtag_score = sum([item['score'] for item in hashtag_res]) / len(hashtag_res)
    # print(f"hashtag_score: {hashtag_score}")
    # print(hashtag_res)
    return hashtag_score, hashtag_res

def cal_history_hashtag(s_history_post, t_history_post):
    s_history_post_hashtag = find_hashtags(s_history_post, 'text_trans').tolist()
    t_history_post_hashtag = find_hashtags(t_history_post, 'text_trans').tolist()
    s_history_post_hashtag = [item for sublist in s_history_post_hashtag for item in sublist]
    t_history_post_hashtag = [item for sublist in t_history_post_hashtag for item in sublist]
    embed_s = model.encode(s_history_post_hashtag, convert_to_tensor=True)
    embed_t = model.encode(t_history_post_hashtag, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embed_s, embed_t)
    cosine_scores = cosine_scores.cpu().numpy()
    s_his_rel_tag = []
    t_his_rel_tag = []
    # 保留相似度大于0.5的
    for i in range(len(s_history_post_hashtag)):
        if cosine_scores[i][np.argmax(cosine_scores[i])] > 0.5:
            s_his_rel_tag.append(s_history_post_hashtag[i])
            t_his_rel_tag.append(t_history_post_hashtag[np.argmax(cosine_scores[i])])
    return s_his_rel_tag, t_his_rel_tag


def find_hashtags(df, col='text_trans'):
    return df[col].str.findall(r'#\w+#|\B#\w+\b')


def find_posts_with_url(df, col='text'):
    return df[df[col].str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',na=False)]


def find_same_url(df, col='text'):
    url_list = []
    post_id_list = []
    for index, row in df.iterrows():
        url = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', row[col])
        if url:
            for u in url:
                url_list.append(u)
                post_id_list.append(row['post_id'])
    return url_list, post_id_list


def find_posts_mention_other_platform(df, platform, col='text'):
    return df[df[col].str.contains(platform, na=False)]

def direct_refer(df, platform,col='text_trans'):
    if platform == 'weibo':
        refer_list = ['weibo', 'Weibo','Chinese Twitter']
    elif platform == 'twitter':
        refer_list = ['twitter', 'Twitter','blue bird']
    else:
        refer_list = ['facebook', 'Facebook']
    df_refer = df[df[col].str.contains('|'.join(refer_list), na=False)]
    refer_num = df_refer.shape[0]
    spread_list = ["source", "cr.", "original post", "source:", "via", "via:", "original link", "credit:", "from",
                    "from:","original link","courtesy of","reprinted from","quote form","cited from","hat tip","originally published by"]
    df_spread = df[df[col].str.contains('|'.join(spread_list), na=False)]
    spread_num = df_spread.shape[0]
    # 加权求和
    refer_score = (0.5 * refer_num + 0.5 * spread_num) / 5
    return refer_score
def find_posts_with_engagement(df, threshold, col=["cnt_retweet", "cnt_agree", "cnt_comment"]):
    return df[df[col] > threshold]


def cal_cluster_factor(A, B, df_data, date, cycle, all_posts, all_users,debug = False):
    A_posts = df_data[df_data['from'] == A]
    B_posts = df_data[df_data['from'] == B]
    end_time = pd.to_datetime(date).strftime('%Y-%m-%d')
    start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
    start_time = start_time.strftime('%Y-%m-%d')
    start_time_half = pd.to_datetime(date) - pd.Timedelta(days=cycle / 2)
    start_time_half = start_time_half.strftime('%Y-%m-%d')
    B_posts = B_posts[(B_posts['publish_time'] >= start_time_half) & (B_posts['publish_time'] <= end_time)]
    # print(f"A: {A_posts.shape[0]} ,B: {B_posts.shape[0]}")
    ##############计算sal_factor######################
    all_A_posts = all_posts[all_posts['from'] == A]
    all_A_posts = all_A_posts[(all_A_posts['publish_time'] >= start_time) & (all_A_posts['publish_time'] <= end_time)]
    all_B_posts = all_posts[all_posts['from'] == B]
    all_B_posts = all_B_posts[(all_B_posts['publish_time'] >= start_time_half) & (all_B_posts['publish_time'] <= end_time)]
    # print(f"all_A: {all_A_posts.shape[0]} ,all_B: {all_B_posts.shape[0]}")
    # 计算salience factor
    if all_A_posts.shape[0] == 0 or all_B_posts.shape[0] == 0:
        sal_factor = 0
    else:
        sal_factor = (A_posts.shape[0] * B_posts.shape[0]) / (all_A_posts.shape[0] * all_B_posts.shape[0])
    # print(f"sal_factor: {sal_factor}")
    ##############计算A B的文本RW_SCORE #######################
    if A_posts.shape[0] == 0 or B_posts.shape[0] == 0:
        rw_score = 0
        rw_res = []
    else:
        rw_score, rw_res = cal_RW_SCORE(A_posts, B_posts)
    # print(f"rw_score: {rw_score}")
    ##############计算A B的文本Hashtag_score#######################
    if A_posts.shape[0] == 0 or B_posts.shape[0] == 0:
        hashtag_score = 0
        hashtag_res = []
    else:
        hashtag_score, hashtag_res = cal_hashtag_score(A_posts, B_posts)
    # print(f"hashtag_score: {hashtag_score}")
    ##############计算A B的SameURL #######################
    A_posts_url = find_posts_with_url(A_posts)
    B_posts_url = find_posts_with_url(B_posts)
    A_posts_url_list,_ = find_same_url(A_posts_url)
    B_posts_url_list,_ = find_same_url(B_posts_url)
    same_url = list(set(A_posts_url_list).intersection(set(B_posts_url_list)))
    same_url_count = len(same_url)
    # print(f"same_url_count: {same_url_count}")
    ##############计算A B的 DirectURL #######################
    A_direct_url = A_posts['url'].dropna().tolist()
    B_direct_url = B_posts['url'].dropna().tolist()
    direct_url = list(set(A_direct_url).intersection(set(B_direct_url)))
    direct_url_count = len(direct_url)
    # print(f"direct_url_count: {direct_url_count}")
    ##############计算A B的 Refer #######################
    direct_refer_score = direct_refer(B_posts, A)
    ##############计算A B的 KOL Inf #######################
    # 注意只计算A的KOL
    Inf_posts = find_posts_with_engagement(A_posts, 100)
    Inf_posts_num = Inf_posts.shape[0]
    # 找出A_posts中的账号
    A_user = A_posts['user_id'].drop_duplicates().tolist()
    # 在self.all_users中找出这些账号
    A_user_info = all_users[all_users['user_id'].isin(A_user)]
    A_user_info['fan'] = A_user_info['fan'].astype(int)
    # 找出粉丝数量大于500的账号
    InfAccts = A_user_info[A_user_info['fan'] > 500].shape[0]
    max_Inf_post = find_posts_with_engagement(all_A_posts, 100)
    max_Inf_post_num = max_Inf_post.shape[0]
    all_A_user = all_A_posts['user_id'].drop_duplicates().tolist()
    all_A_user_info = all_users[all_users['user_id'].isin(all_A_user)]
    all_A_user_info['fan'] = all_A_user_info['fan'].astype(int)
    max_InfAccts = all_A_user_info[all_A_user_info['fan'] > 500].shape[0]
    # print(f"A engagement: {Inf_posts_num}, fan > 500: {InfAccts}, max_Inf_post: {max_Inf_post_num}, max_fan > 500: {max_InfAccts}")
    inf_post = 0 if (max_Inf_post_num == 0) else (Inf_posts_num / max_Inf_post_num)
    inf_acct = 0 if (max_InfAccts == 0) else (InfAccts / max_InfAccts)
    KOL_inf = inf_post + inf_acct
    # print(f"KOL_inf: {KOL_inf}")
    #######################################################
    sim_con = 0.1 * rw_score + 0.3 * same_url_count + 0.5 * direct_url_count + 0.4 * direct_refer_score + 0.1 * hashtag_score
    # 计算指数
    p = 1 - np.exp(-1 * sal_factor - KOL_inf * sim_con)
    # print(f"p: {p}")
    def print_data(debug = False):
        if debug:
            print(f"A: {A_posts.shape[0]} ,B: {B_posts.shape[0]}")
            print(f"all_A: {all_A_posts.shape[0]} ,all_B: {all_B_posts.shape[0]}")
            print(f"sal_factor: {sal_factor}")
            print(f"rw_score: {rw_score}")
            print(f"hashtag_score: {hashtag_score}")
            print(f"same_url_count: {same_url_count}")
            print(f"direct_url_count: {direct_url_count}")
            print(f"direct_refer_score: {direct_refer_score}")
            print(f"A engagement: {Inf_posts_num}, fan > 500: {InfAccts}, max_Inf_post: {max_Inf_post_num}, max_fan > 500: {max_InfAccts}")
            print(f"KOL_inf: {KOL_inf}")
            print(f"p: {p}")
    print_data(debug)
    return p, rw_res, hashtag_res
