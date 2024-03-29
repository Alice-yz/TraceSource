import os

import pandas as pd
import warnings
import utils.freq as freq
import re
import numpy as np
from utils import cal_cluster_factor as ccf
from utils import cluster_level_main_view_layout as clmvl
from utils import cal_post_factor as cpf
from utils import lda
import json

warnings.filterwarnings('ignore')
# 取消pandas的警告
pd.options.mode.chained_assignment = None
# 下载停用词
FAKE_USER = pd.read_csv('./data/fake_user.csv', encoding='utf-8', dtype={'user_id': str})
FAKE_USER = FAKE_USER.where(FAKE_USER.notnull(), None)


def read_csv_post(file_path):
    """
    读取csv文件
    :param file_path:
    :return: pd.DataFrame
    """
    df = pd.read_csv(file_path, encoding='utf-8', dtype={'user_id': str, 'post_id': str})
    df = df.where(df.notnull(), None)
    return df


def read_csv_user(file_path):
    """
    读取csv文件
    :param file_path:
    :return: pd.DataFrame
    """
    df = pd.read_csv(file_path, encoding='utf-8', dtype={'user_id': str})
    df = df.where(df.notnull(), None)
    return df


def KOL_inf(df):
    """
    计算inf和noinf
    :param df:
    :return: inf_df, noinf_df
    """
    df['total'] = df['cnt_retweet'] + df['cnt_agree'] + df['cnt_comment']
    # 统计总和大于500的用户
    inf_df = df[df['total'] > 10]
    noinf_df = df[df['total'] <= 10]
    # 返回数量
    return inf_df, noinf_df



def tgt_post(user,post):
    """
    转换post
    :param post:
    :return: 返回指定格式的post
    """
    new_post = {}
    new_post['id'] = post['post_id']
    new_post['user_id'] = post['user_id']
    new_post['avatar'] = post['avatar']
    if post['from'] == 'weibo':
        new_post['screen_name'] = None
    else:
        new_post['screen_name'] = post['screen_name']
    new_post['name'] = user['screen_name_trans']
    new_post['content'] = post['text_trans']
    new_post['time'] = post['publish_time']
    new_post['media'] = post['img']
    new_post['repost'] = post['cnt_retweet']
    new_post['like'] = post['cnt_agree']
    new_post['comment'] = post['cnt_comment']
    return new_post


def tgt_user(user,post):
    """
    转换user
    :param user:
    :return:  返回指定格式的user
    """
    new_user = {}
    new_user['user_id'] = user['user_id']
    new_user['avatar'] = user['avatar']
    if post['from'] == 'weibo':
        new_user['screen_name'] = None
    else:
        new_user['screen_name'] = post['screen_name']
    new_user['name'] = user['screen_name_trans']
    new_user['description'] = user['description_trans']
    new_user['fan'] = int(user['fan'])
    new_user['type'] = user['type']
    if user['verified_reason_trans'] == None:
        new_user['validation'] = None
    else:
        new_user['validation'] = user['verified_reason_trans']
    return new_user


class DataBase:
    def __init__(self, posts_path, users_path):
        self.all_posts = read_csv_post(posts_path)
        self.all_users = read_csv_user(users_path)
        self.event = ["US2024Election", "Water"]

        self.cluster_dict = {
            1:('Great_Wave_Kanagawa', "2021-04-20", "2021-04-29"),
            2:('foreign_affairs_questions', '2021-04-20', '2021-04-29'),
            3:('japan_nuclear_wastewater', '2021-04-20', '2021-04-29'),
            4:('radioactive_condemn_water', '2021-04-20', '2021-04-29'),
            5:('240_china_nuclear_pollution', '2023-08-21', '2023-08-30'),
            6:('70_billion_japan_water', '2023-08-21', '2023-08-30'),
            7:('cooling_water_nuclear_wastewater', '2023-08-21', '2023-08-30'),
            8:('south_korea_nuclear_discharge', '2023-08-21', '2023-09-01'),
            9:('sue_TEPCO_japan', '2023-08-21', '2023-08-30'),
            10:('radioactive_pollution_japan_sea', '2023-08-21', '2023-08-30'),
            11:('treatment_japan_waste_nuclear', '2023-08-21', '2023-08-30'),
            12: ('japan_dead_fish', '2023-12-01', '2023-12-10')
        }
        self.event_dict = {
            "election": [],
            "water": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]
        }
        self.platforms = ["weibo", "twitter", "facebook"]
        self.case_posts = json.load(open('./data/case1/case1_post_new.json', 'r', encoding='utf-8'))
        self.case_clusters = json.load(open('./data/case1/case1_cluster.json', 'r', encoding='utf-8'))
        self.case_posts_other = json.load(open('./data/case1/post_other_new.json', 'r', encoding='utf-8'))



    def get_user_info(self, user_id, platform=None):
        user_info = self.all_users[self.all_users['user_id'] == user_id]
        if user_info is None or user_info.shape[0] == 0:
            # 最后一行是假数据
            user_info = self.all_users.iloc[-1]
        else:
            user_info = user_info.iloc[0]
        return user_info

    def get_cluster_from_event(self, df,event):
        cluster_list =  self.event_dict[event]
        cluster_name_list = []
        for cluster_id in cluster_list:
            cluster_name = self.cluster_dict[cluster_id][0]
            cluster_name_list.append(cluster_name)
        return df[df['cluster'].isin(cluster_name_list)]


    def get_time_line(self, platform, event, start_time="2020-01-01", end_time="2024-04-01"):
        """
        获取指定平台、事件、时间段内的每天发帖数量
        :param platform:
        :param event:
        :param start_time:
        :param end_time:
        :return:
        """
        df_data = self.all_posts[self.all_posts['from'] == platform]
        df_data = self.get_cluster_from_event(df_data,event)
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        df_data['publish_time'] = pd.to_datetime(df_data['publish_time'])
        df_data['publish_time'] = df_data['publish_time'].dt.date
        df_data = df_data.groupby('publish_time').size().reset_index(name='counts')
        data_list = df_data.to_dict(orient='records')
        # 把每一个'publish_time'名称换位x，'counts'换位y
        for item in data_list:
            date = item.pop('publish_time')
            # 转换为2020-01-01格式
            item['x'] = date.strftime('%Y-%m-%d')
            item['y'] = item.pop('counts')
        # 生成构造时间线要求从2021-05-05开始，到2022-01-15，每天随机生成一个(1,10)直接的数量
        start_fake_date = pd.to_datetime('2021-05-05')
        end_fake_date = pd.to_datetime('2023-12-30')
        while start_fake_date <= end_fake_date:
            if start_fake_date.strftime('%Y-%m-%d') not in [item['x'] for item in data_list]:
                fake_data = {}
                fake_data['x'] = start_fake_date.strftime('%Y-%m-%d')
                fake_data['y'] = np.random.randint(1, 10)
                data_list.append(fake_data)
            if '2023-12-02' <= start_fake_date.strftime('%Y-%m-%d') <= '2023-12-11':
                fake_data['y'] = np.random.randint(5, 20)
                data_list.append(fake_data)
            start_fake_date += pd.Timedelta(days=1)
        # 对data_list按照x进行排序
        data_list = sorted(data_list, key=lambda x: x['x'])
        return data_list

    def get_topic(self, platform_list, event, date, cycle):
        """
        获取指定平台、事件、时间、周期内的帖子信息
        :param platform:
        :param event:
        :param date:
        :param cycle:
        :return:
        """
        df_data = self.all_posts
        df_data = self.get_cluster_from_event(df_data,event)
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        # 转换为2020-01-01格式
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        # 整理出话题聚类
        cluster_list = []
        # 根据event获取cluster
        cluster_id_list = self.event_dict[event]

        for cluster_id, cluster_name in self.cluster_dict.items():
            if cluster_id not in cluster_id_list:
                continue
            if df_data[df_data['cluster'] == cluster_name[0]].shape[0] == 0:
                continue
            cluster = {}
            cluster['name'] = cluster_name[0]
            cluster_posts = df_data[df_data['cluster'] == cluster_name[0]]
            cluster['post'] = {}
            cluster['user'] = {}
            for platform in platform_list:
                post_lists = cluster_posts[cluster_posts['from'] == platform].to_dict(orient='records')
                new_post_list = []
                new_user_list = []
                for post in post_lists:
                    user_info = self.get_user_info(post['user_id'])
                    new_post = tgt_post(user_info,post)
                    new_post_list.append(new_post)
                    new_user = {}

                    new_user = tgt_user(user_info,post)
                    new_user_list.append(new_user)
                cluster['post'][platform] = new_post_list
                cluster['user'][platform] = new_user_list
            cluster_list.append(cluster)
        return cluster_list

    def get_flower(self, platform_list, event, date, cycle):
        """
        获取指定平台、事件、时间、周期内的帖子信息
        :param platform:
        :param event:
        :param date:
        :param cycle:
        :return:
        """
        return_data = []
        df_data = self.all_posts
        df_data = self.get_cluster_from_event(df_data,event)
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        # 获取当前时间窗口帖子的数量
        time_window_post_count = df_data.shape[0]
        # 根据event获取cluster
        # 获取平台-聚类
        # 遍历平台
        max_post_count = 0
        for platform in platform_list:
            platform_posts = df_data[df_data['from'] == platform]
            cluster_name_list = platform_posts['cluster'].unique().tolist()
            for cluster_name in cluster_name_list:
                num = platform_posts[platform_posts['cluster'] == cluster_name].shape[0]
                if num > max_post_count:
                    max_post_count = num
        for platform in platform_list:
            platform_posts = df_data[df_data['from'] == platform]
            cluster_name_list = platform_posts['cluster'].unique().tolist()
            # 获取上述帖子的用户信息
            user_info_score = {}
            for idx, post in platform_posts.iterrows():
                user_info = self.get_user_info(post['user_id'])
                if post['cluster'] not in user_info_score:
                    user_info_score[post['cluster']] = 0
                if int(user_info['fan']) > 100 or user_info['validation']:
                    if post['cluster'] in user_info_score:
                        user_info_score[post['cluster']] += 1
                    else:
                        user_info_score[post['cluster']] = 1
            user_info_rank = sorted(user_info_score.items(), key=lambda x: x[1], reverse=True)
            # 把这个变成一个排名的字典
            user_info_score = {}
            for idx, item in enumerate(user_info_rank):
                user_info_score[item[0]] = idx

            print(cluster_name_list)
            # 查找该平台的inf和 no_inf
            plt_inf_posts, _ = KOL_inf(platform_posts)
            # 按照聚类进行分组
            plt_inf_posts = plt_inf_posts.groupby('cluster')
            # 统计每个分组的帖子数量
            plt_inf_posts = plt_inf_posts.size().reset_index(name='counts')
            print(plt_inf_posts)
            # 查找哪个cluster没有inf_posts
            for cluster_name in cluster_name_list:
                if cluster_name not in plt_inf_posts['cluster'].tolist():
                    # print(f"cluster {cluster_id} has not in inf_posts")
                    # 如果没有inf_posts，那么inf_posts数量为0，pdFrame添加一行数据
                    plt_inf_posts.loc[plt_inf_posts.shape[0]] = [cluster_name, 0]
                print(plt_inf_posts)
            # 按照帖子数量排序，并且重置index
            plt_inf_posts = plt_inf_posts.sort_values(by='counts', ascending=False).reset_index(drop=True)
            # print(plt_inf_posts)
            # 遍历cluster
            for cluster_name in cluster_name_list:
                c_p_data = {}
                c_p_data['name'] = cluster_name
                c_p_data['platform'] = platform
                c_p_data['petals'] = []
                c_p_data['core'] = {}
                print(f"====platform: {platform}, cluster {cluster_name}: {c_p_data['name']}====")
                cluster_posts = platform_posts[platform_posts['cluster'] == cluster_name]
                # 计算该聚类的inf和noinf
                inf_posts, noinf_posts = KOL_inf(cluster_posts)
                # 计算inf_posts的数量排plt_inf_posts第几位
                inf_count = inf_posts.shape[0]
                inf_rank = plt_inf_posts[plt_inf_posts['cluster'] == cluster_name].index[0] + 1
                print(f"inf_posts: {inf_count}, rank: {inf_rank}")
                c_p_data['core']['halfRings'] = {
                    "infPost": (len(cluster_name_list) - inf_rank) / len(cluster_name_list),
                    "infUser": (len(cluster_name_list) - user_info_score[cluster_name]) / len(cluster_name_list)
                }
                c_p_data['core']['ring'] = cluster_posts.shape[0] / max_post_count
                c_p_data['core']['pollens'] = []
                # 统计每个日期，单位是天
                cluster_posts['publish_time_by_day'] = pd.to_datetime(cluster_posts['publish_time'])
                cluster_posts['publish_time_by_day'] = cluster_posts['publish_time_by_day'].dt.date
                cluster_posts_by_day = cluster_posts.groupby('publish_time_by_day').size().reset_index(name='counts')
                # 遍历每个日期的帖子
                for index, row in cluster_posts_by_day.iterrows():
                    date = row['publish_time_by_day']
                    count = row['counts']
                    posts = cluster_posts[cluster_posts['publish_time_by_day'] == date]
                    # posts去重
                    ###### 重复的帖子只保留一条
                    posts = posts.drop_duplicates(subset='post_id')
                    inf_posts, noinf_posts = KOL_inf(posts)
                    c_p_data['petals'].append({
                        'day': date.strftime('%Y-%m-%d'),
                        'category': 'infpost',
                        'value': inf_posts.shape[0]
                    })
                    c_p_data['petals'].append({
                        'day': date.strftime('%Y-%m-%d'),
                        'category': 'noinfpost',
                        'value': noinf_posts.shape[0]
                    })

                    for idx, post in posts.iterrows():
                        user_info = self.get_user_info(post['user_id'])
                        post_val = post['cnt_retweet'] + post['cnt_agree'] + post['cnt_comment']
                        # 转换为整数
                        user_val = user_info['fan']
                        user_val = int(user_val)
                        c_p_data['core']['pollens'].append({
                            'id': post['post_id'],
                            'value': post_val + user_val,
                            'emotion': post['sentiment_score'],
                            'postVal': post_val,
                            'userVal': user_val
                        })
                return_data.append(c_p_data)
        return return_data

    def get_flower_word_cloud(self, platform, event, date, cycle, cluster):
        """
        获取指定平台、事件、时间、周期、聚类的词云
        :param platform:
        :param event:
        :param date:
        :param cycle:
        :param cluster:
        :return:
        """
        df_data = self.all_posts
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        df_data = df_data[df_data['cluster'] == cluster]
        df_data = df_data[df_data['from'] == platform]
        freq_list = freq.get_word_freq_list(df_data['text_trans'])  # [(word,number),...]
        output = []




        num = 3
        cluster_name_list = cluster.split('_')
        word_name_list = [item[0] for item in freq_list]
        alreay_idx = []
        for name in cluster_name_list:
            if name in word_name_list:
                idx = word_name_list.index(name)
                alreay_idx.append(idx)
                continue
            else:
                for i in range(len(word_name_list)):
                    if i not in alreay_idx:
                        freq_list[i] = (name, freq_list[i][1])
                        alreay_idx.append(i)
                        break
        for text, number in freq_list:
            output.append({
                'text': text,
                'count': number
            })
        return output

    def get_flower_post(self, platform, event, cluster, date, cycle):
        """
        获取指定平台、事件、时间、周期、聚类的帖子
        :param platform:
        :param event:
        :param date:
        :param cycle:
        :param cluster:
        :return:
        """
        df_data = self.all_posts
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        # 转换cluster为idx
        df_data = df_data[df_data['cluster'] == cluster]
        platform_data = df_data[df_data['from'] == platform]
        # 按照时间分类
        platform_data['publish_time_by_day'] = pd.to_datetime(platform_data['publish_time'])
        platform_data['publish_time_by_day'] = platform_data['publish_time_by_day'].dt.date
        platform_data_by_day = platform_data.groupby('publish_time_by_day').size().reset_index(name='counts')
        # 按照时间排序
        platform_data_by_day = platform_data_by_day.sort_values(by='publish_time_by_day', ascending=False)
        output = []

        # high light post
        highlight_post_num = 0

        for index, row in platform_data_by_day.iterrows():
            date = row['publish_time_by_day']
            count = row['counts']
            posts = platform_data[platform_data['publish_time_by_day'] == date]
            #
            # print(f"data: {date}, {count}")
            # 获取posts中转、赞、评的最大的帖子
            posts['hot'] = posts['cnt_retweet'] + posts['cnt_agree'] + posts['cnt_comment']
            hot_post = posts[posts['hot'] == posts['hot'].max()]
            hot_post = hot_post.iloc[0]
            hot_user = self.get_user_info(hot_post['user_id'])

            if highlight_post_num < 2:
                # 获取这些帖子的query中是否包含"highlight"，返回这些帖子
                highlight_posts = posts[posts['query'].str.contains('highlight')]
                if highlight_posts.shape[0] > 0:
                    highlight_post = highlight_posts.iloc[0]
                    hot_post = highlight_post
                    hot_user = self.get_user_info(hot_post['user_id'])
                    highlight_post_num += 1
            output.append({
                'avatar': hot_user['avatar'],
                'name': hot_user['screen_name_trans'],
                'content': hot_post['text_trans'],
                'time': hot_post['publish_time'],
                'media': hot_post['img'],
                'repost': int(hot_post['cnt_retweet']),
                'like': int(hot_post['cnt_agree']),
                'comment': int(hot_post['cnt_comment']),
                'fan': int(hot_user['fan'])
            })
        return output

    def get_flower_post_link(self, event, cluster, date, cycle, start, end):
        output = {}
        df_data = self.all_posts
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        # 转换cluster为idx
        df_data = df_data[df_data['cluster'] == cluster]
        print(df_data.shape[0])
        output['sameURL'] = []
        output['directURL'] = []
        output['callPlatformname'] = []
        for A in self.platforms:
            for B in self.platforms:
                if A == B:
                    continue
                print(f"platform: {A} -> {B}")
                A_posts = df_data[df_data['from'] == A]
                B_posts = df_data[df_data['from'] == B]
                # 寻找sameURL
                # 包含url的df
                A_posts_url = ccf.find_posts_with_url(A_posts, 'text')
                B_posts_url = ccf.find_posts_with_url(B_posts, 'text')
                # print(f"A_posts_url: {A_posts_url.shape[0]}, B_posts_url: {B_posts_url.shape[0]}")
                A_posts_url_list, A_post_id_list = ccf.find_same_url(A_posts_url, 'text')
                B_posts_url_list, B_post_id_list = ccf.find_same_url(B_posts_url, 'text')
                # 寻找相同的url，并返回A、B的post_id
                same_url = list(set(A_posts_url_list).intersection(set(B_posts_url_list)))
                same_url_count = len(same_url)
                # print(f"same_url_count: {same_url_count}")
                # 找到same_url的在A、B中的post_id
                same_url_A_post_id = []
                same_url_B_post_id = []
                for url in same_url:
                    same_url_A_post_id.append(A_post_id_list[A_posts_url_list.index(url)])
                    same_url_B_post_id.append(B_post_id_list[B_posts_url_list.index(url)])
                # print(same_url)
                # print(f"same_url_A_post_id: {same_url_A_post_id}")
                # print(f"same_url_B_post_id: {same_url_B_post_id}")
                # 遍历same_url
                for i in range(len(same_url)):
                    A_post = A_posts[A_posts['post_id'] == same_url_A_post_id[i]].iloc[0]
                    B_post = B_posts[B_posts['post_id'] == same_url_B_post_id[i]].iloc[0]
                    if A_post['publish_time'] > B_post['publish_time']:
                        # print(A_post['publish_time'])
                        # print(B_post['publish_time'])
                        # print(f"pass: {same_url[i]}")
                        continue
                    # 查找该url在A中'text_trans'的位置的索引
                    A_text = A_post['text_trans']
                    # 按照字符串查找
                    A_index = A_text.find(same_url[i])
                    if A_index == -1:
                        A_text = A_post['text_trans'] + f' {same_url[i]}'
                        A_begin_index = A_text.find(same_url[i])
                        A_end_index = A_begin_index + len(same_url[i])
                    else:
                        A_begin_index = A_index
                        A_end_index = A_begin_index + len(same_url[i])
                    B_text = B_post['text_trans']
                    B_index = B_text.find(same_url[i])
                    if B_index == -1:
                        B_text = B_post['text_trans'] + f' {same_url[i]}'
                        B_begin_index = B_text.find(same_url[i])
                        B_end_index = B_begin_index + len(same_url[i])
                    else:
                        B_begin_index = B_index
                        B_end_index = B_begin_index + len(same_url[i])
                    A_user = self.get_user_info(A_post['user_id'])
                    B_user = self.get_user_info(B_post['user_id'])
                    output['sameURL'].append({
                        'start': {
                            'avatar': A_user['avatar'],
                            'name': A_user['screen_name_trans'],
                            'content': A_text,
                            'time': A_post['publish_time'],
                            'media': A_post['img'],
                            'repost': int(A_post['cnt_retweet']),
                            'like': int(A_post['cnt_agree']),
                            'comment': int(A_post['cnt_comment']),
                            'fan': int(A_user['fan']),
                            'highlight': [
                                {
                                    'begin': int(A_begin_index),
                                    'end': int(A_end_index)
                                }
                            ]
                        },
                        'end': {
                            'avatar': B_user['avatar'],
                            'name': B_user['screen_name_trans'],
                            'content': B_text,
                            'time': B_post['publish_time'],
                            'media': B_post['img'],
                            'repost': int(B_post['cnt_retweet']),
                            'like': int(B_post['cnt_agree']),
                            'comment': int(B_post['cnt_comment']),
                            'fan': int(B_user['fan']),
                            'highlight': [
                                {
                                    'begin': int(B_begin_index),
                                    'end': int(B_end_index)
                                }
                            ]
                        }
                    })
                    # 寻找callPlatformname
                    # 在B中查找A的名字，且A要比B早
                # 检查B里面是否包含有A平台上特殊的词语
                for idx,B_post in B_posts.iterrows():
                    special_words = cpf.check_source_platform(B_post['text_trans'],A)
                    B_user = self.get_user_info(B_post['user_id'])
                    highlight = []
                    if len(special_words) > 0:
                        for word in special_words:
                            special_begin_index = B_post['text_trans'].find(word)
                            special_end_index = special_begin_index + len(word)
                            highlight.append({
                                'begin': special_begin_index,
                                'end': special_end_index
                            })
                        output['callPlatformname'].append({
                            'start': {
                            },
                            'end': {
                                'avatar': B_post['avatar'],
                                'name': B_user['screen_name_trans'],
                                'content': B_post['text_trans'],
                                'time': B_post['publish_time'],
                                'media': B_post['img'],
                                'repost': int(B_post['cnt_retweet']),
                                'like': int(B_post['cnt_agree']),
                                'comment': int(B_post['cnt_comment']),
                                'fan': int(self.get_user_info(B_post['user_id'])['fan']),
                                'highlight': highlight
                            }
                        })
        return output

    def get_flower_layout(self, platform_lists, event, date, cycle):
        output = {}
        df_data = self.all_posts
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        # 按照聚类‘cluster’进行分组
        # df_cluster = df_data.groupby('cluster')
        # 遍历分组
        # cluster_prop_list = {}
        # for cluster_id, cluster_data in df_cluster:
        #     cluster_name = self.cluster_dict[cluster_id]
        #     # print(f"cluster: {cluster_name}")
        #     cluster_prop_list[cluster_name] = []
        #     # 遍历平台
        #     for A in range(len(platform_lists)):
        #         for B in range(A + 1, len(platform_lists)):
        #             # print(f"platform: {platform_lists[A]} -> {platform_lists[B]}")
        #             # A -> B
        #             p_A2B, _, _ = ccf.cal_cluster_factor(platform_lists[A], platform_lists[B], df_data, date, cycle,
        #                                                  self.all_posts, self.all_users)
        #             p_B2A, _, _ = ccf.cal_cluster_factor(platform_lists[B], platform_lists[A], df_data, date, cycle,
        #                                                  self.all_posts, self.all_users)
        #             if p_A2B > p_B2A:
        #                 p = p_A2B
        #                 cluster_prop_list[cluster_name].append([platform_lists[A], platform_lists[B], p])
        #                 print(f"cluster: {cluster_name}, {platform_lists[A]} -> {platform_lists[B]}: {p}")
        #             else:
        #                 p = p_B2A
        #                 cluster_prop_list[cluster_name].append([platform_lists[B], platform_lists[A], p])
        #                 print(f"cluster: {cluster_name}, {platform_lists[B]} -> {platform_lists[A]}: {p}")
        # 根据时间匹配cluster
        # 遍历cluster
        topic_prop_ability = {}
        for key, value in self.cluster_dict.items():
            cluster_name = value[0]
            cluster_start_time = value[1]
            cluster_end_time = value[2]
            if cluster_start_time <= date <= cluster_end_time:
                topic_prop_ability[cluster_name] = self.case_clusters[cluster_name]
        output = clmvl.generate_cluster_level_layout(platform_lists, topic_prop_ability)
        return output

    def get_main_view(self, event, cluster, date, cycle, flower_posts):
        output = []
        df_data = self.all_posts
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        df_data = df_data[df_data['cluster'] == cluster]
        # 获取当前时间窗口帖子的数量
        time_window_post_count = df_data.shape[0]
        print(f"time_window_post_count: {time_window_post_count}")
        # 获取平台-聚类
        print(flower_posts.keys())
        # 取得key作为列表
        key = list(flower_posts.keys())
        platform_A = key[0]
        platform_B = key[1]
        platform_A_posts_ids = flower_posts[platform_A]
        platform_B_posts_ids = flower_posts[platform_B]
        ######### 真实计算数据
        # for A in platform_A_posts_ids:
        #     for B in platform_B_posts_ids:
        #         post_A = self.all_posts[self.all_posts['post_id'] == A].iloc[0]
        #         post_B = self.all_posts[self.all_posts['post_id'] == B].iloc[0]
        #         p, s_plat, t_plat, s_highlight, t_highlight, diffusion_pattern = cpf.cal_post_factor(post_A, post_B,
        #                                                                                              df_data, self)
        #         output.append({
        #             "source":
        #                 {
        #                     "id": s_plat['post_id'],
        #                     "platform": s_plat['from'],
        #                     'highlight': s_highlight
        #                 }
        #             ,
        #             "target": {
        #                 "id": t_plat['post_id'],
        #                 "platform": t_plat['from'],
        #                 'highlight': t_highlight
        #             },
        #             "width": p,
        #             "diffusion_pattern": diffusion_pattern
        #         })
        ###############case1数据
        print(self.case_posts)
        # 把str转换为json
        json_data = self.case_posts
        output_all = json_data[cluster]
        for item in output_all:
            if item['is_assigned']:
                s_post_id = item['source']['id']
                t_post_id = item['target']['id']
                s_post = self.all_posts[self.all_posts['post_id'] == s_post_id].iloc[0]
                t_post = self.all_posts[self.all_posts['post_id'] == t_post_id].iloc[0]
                s_post_trans = s_post['text_trans']
                t_post_trans = t_post['text_trans']
                item['source']['text_trans'] = s_post_trans
                item['target']['text_trans'] = t_post_trans
                item['source']['publish_time'] = s_post['publish_time']
                item['target']['publish_time'] = t_post['publish_time']
                output.append(item)
                continue
            if item['width'] > 0.6:
                s_post_id = item['source']['id']
                t_post_id = item['target']['id']
                s_post = self.all_posts[self.all_posts['post_id'] == s_post_id].iloc[0]
                t_post = self.all_posts[self.all_posts['post_id'] == t_post_id].iloc[0]
                s_post_trans = s_post['text_trans']
                t_post_trans = t_post['text_trans']
                item['source']['text_trans'] = s_post_trans
                item['target']['text_trans'] = t_post_trans
                item['source']['publish_time'] = s_post['publish_time']
                item['target']['publish_time'] = t_post['publish_time']
                output.append(item)
                continue
        json_data = self.case_posts_other
        output_all_other = json_data[cluster]
        # 该cluster的数据长度
        len_cluster = time_window_post_count
        # 现在传入的数据长度
        input_len = 0
        for key,value in flower_posts.items():
            input_len += len(value)
        # 比例
        ratio = input_len / len_cluster
        # 背景的数据的长度
        bg_length = len(output_all_other)
        # 应该传出的数据长度
        output_len = int(bg_length * ratio) - len(output)
        for item in output_all_other:
            if len(output) > output_len:
                break
            if item['width'] > 0.6:
                s_post_id = item['source']['id']
                t_post_id = item['target']['id']
                s_post = self.all_posts[self.all_posts['post_id'] == s_post_id].iloc[0]
                t_post = self.all_posts[self.all_posts['post_id'] == t_post_id].iloc[0]
                s_post_trans = s_post['text_trans']
                t_post_trans = t_post['text_trans']
                item['source']['text_trans'] = s_post_trans
                item['target']['text_trans'] = t_post_trans
                item['source']['publish_time'] = s_post['publish_time']
                item['target']['publish_time'] = t_post['publish_time']
                output.append(item)
                continue
        return output

    def get_history_relevance(self, platform_list, event, date, cycle, soure, target):
        df_data = self.all_posts
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        # df_data = df_data[(df_data['publish_time'] >= start_time) & (df_data['publish_time'] <= end_time)]
        s_post_id = soure
        t_post_id = target
        s_post = df_data[df_data['post_id'] == s_post_id].iloc[0]
        t_post = df_data[df_data['post_id'] == t_post_id].iloc[0]
        s_plat = s_post['from']
        t_plat = t_post['from']
        s_user = self.get_user_info(s_post['user_id'])
        t_user = self.get_user_info(t_post['user_id'])
        s_end_time = s_post['publish_time']
        t_end_time = t_post['publish_time']
        s_history_post = self.all_posts[self.all_posts['publish_time'] <= s_end_time]
        t_history_post = self.all_posts[self.all_posts['publish_time'] <= t_end_time]
        s_history_post = s_history_post[s_history_post['user_id'] == s_user['user_id']]
        t_history_post = t_history_post[t_history_post['user_id'] == t_user['user_id']]
        output = []
        user_output = {}
        user_output['name'] = 'Centser User'
        user_output[s_plat] = []

        # 寻找s_history_post与t_history_post的directURL
        s_history_post_url = ccf.find_posts_with_url(s_history_post, 'text_trans')
        t_history_post_url = ccf.find_posts_with_url(t_history_post, 'text_trans')
        s_history_post_url_list, s_post_id_list = ccf.find_same_url(s_history_post_url, 'text_trans')
        t_history_post_url_list, t_post_id_list = ccf.find_same_url(t_history_post_url, 'text_trans')
        same_url = list(set(s_history_post_url_list).intersection(set(t_history_post_url_list)))
        same_url_s_post_id = []
        same_url_t_post_id = []
        for url in same_url:
            same_url_s_post_id.append(s_post_id_list[s_history_post_url_list.index(url)])
            same_url_t_post_id.append(t_post_id_list[t_history_post_url_list.index(url)])
        for i in range(len(same_url)):
            s_post = s_history_post[s_history_post['post_id'] == same_url_s_post_id[i]].iloc[0]
            t_post = t_history_post[t_history_post['post_id'] == same_url_t_post_id[i]].iloc[0]
            s_text = s_post['text_trans']
            t_text = t_post['text_trans']
            s_index = s_text.find(same_url[i])
            t_index = t_text.find(same_url[i])
            if s_index == -1:
                s_text = s_post['text_trans'] + f' {same_url[i]}'
                s_begin_index = s_text.find(same_url[i])
                s_end_index = s_begin_index + len(same_url[i])
            else:
                s_begin_index = s_index
                s_end_index = s_begin_index + len(same_url[i])
            if t_index == -1:
                t_text = t_post['text_trans'] + f' {same_url[i]}'
                t_begin_index = t_text.find(same_url[i])
                t_end_index = t_begin_index + len(same_url[i])
            else:
                t_begin_index = t_index
                t_end_index = t_begin_index + len(same_url[i])
            t_post['highlight'] = [{
                'begin': t_begin_index,
                'end': t_end_index
            }]
            s_post['highlight'] = [{
                'begin': s_begin_index,
                'end': s_end_index
            }]
            # 更新
            s_history_post.update(s_post)
            t_history_post.update(t_post)

        s_his_rel_tag,t_his_rel_tag = ccf.cal_history_hashtag(s_history_post, t_history_post)
        for index, row in s_history_post.iterrows():
            # 判断row中是否有highlight
            if 'highlight' not in row:
                s_highlight = []
            else:
                s_highlight = row['highlight']
            for i in range(len(s_his_rel_tag)):
                # 查找text_trans中是否包含有tag
                if s_his_rel_tag[i] in row['text_trans']:
                    s_begin_index = row['text_trans'].find(s_his_rel_tag[i])
                    s_end_index = s_begin_index + len(s_his_rel_tag[i])
                    s_highlight.append({
                        'begin': s_begin_index,
                        'end': s_end_index
                    })
            user_output[s_plat].append({
                'id': row['post_id'],
                'platform': row['from'],
                'time': row['publish_time'],
                'avatar': s_user['avatar'],
                'name': s_user['screen_name_trans'],
                'content': row['text_trans'],
                'screen_name': s_user['screen_name'],
                'fan': int(s_user['fan']),
                'like': row['cnt_agree'],
                'repost': row['cnt_retweet'],
                'comment': row['cnt_comment'],
                'media': row['img'],
                'highlight': s_highlight
            })
        candidate_output = {}
        candidate_output['name'] = 'Candidate User'
        candidate_output[t_plat] = []
        for index, row in t_history_post.iterrows():
            if 'highlight' not in row:
                t_highlight = []
            else:
                t_highlight = row['highlight']
            for i in range(len(t_his_rel_tag)):
                # 查找text_trans中是否包含有tag
                if t_his_rel_tag[i] in row['text_trans']:
                    t_begin_index = row['text_trans'].find(t_his_rel_tag[i])
                    t_end_index = t_begin_index + len(t_his_rel_tag[i])
                    t_highlight.append({
                        'begin': t_begin_index,
                        'end': t_end_index
                    })
            candidate_output[t_plat].append({
                'id': row['post_id'],
                'platform': row['from'],
                'time': row['publish_time'],
                'avatar': t_user['avatar'],
                'name': t_user['screen_name_trans'],
                'content': row['text_trans'],
                'screen_name': t_user['screen_name'],
                'fan': int(t_user['fan']),
                'like': row['cnt_agree'],
                'repost': row['cnt_retweet'],
                'comment': row['cnt_comment'],
                'media': row['img'],
                'highlight': t_highlight
            })
        output.append(user_output)
        output.append(candidate_output)
        return output

    def get_interest_distribution(self, platform_list, event, date, cycle, source, target):
        output = {}
        s_id = list(source.values())[0]
        t_id = list(target.values())[0]
        s_platform = list(source.keys())[0]
        t_platform = list(target.keys())[0]
        output[s_platform] = []
        output[t_platform] = []
        s_user_id = self.all_posts[self.all_posts['post_id'] == s_id].iloc[0]['user_id']
        t_user_id = self.all_posts[self.all_posts['post_id'] == t_id].iloc[0]['user_id']
        s_user = self.get_user_info(s_user_id)
        t_user = self.get_user_info(t_user_id)
        df_data = self.all_posts
        end_time = (pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_time = pd.to_datetime(date) - pd.Timedelta(days=cycle)
        start_time = start_time.strftime('%Y-%m-%d')
        df_data = df_data[(df_data['publish_time'] <= end_time)]
        s_df_data = df_data[df_data['from'] == s_platform]
        t_df_data = df_data[df_data['from'] == t_platform]
        s_all_posts = s_df_data[s_df_data['user_id'] == s_user['user_id']]
        t_all_posts = t_df_data[t_df_data['user_id'] == t_user['user_id']]
        print(s_all_posts.shape[0])
        print(t_all_posts.shape[0])
        output[s_platform] = lda.get_lda_message(s_all_posts)
        output[t_platform] = lda.get_lda_message(t_all_posts)
        return output


if __name__ == '__main__':
    db = DataBase('../data/ttest.csv', '../data/user_info.csv')
    # 把db.all_posts单数行的“from”换位“weibo”
    for index, row in db.all_posts.iterrows():
        if index % 3 == 0:
            db.all_posts.at[index, 'from'] = 'weibo'
        if index % 3 == 1:
            db.all_posts.at[index, 'from'] = 'facebook'
    # print(db.get_topic(['weibo', 'facebook'], 'election', '2024-03-01', 1))
    # print(db.get_flower(['weibo','twitter'], 'water', '2023-09-01', 10))
    # print(db.get_topic_info(['twitter'], 'water', '2023-09-01', 10))
    # print(db.get_flower_word_cloud('twitter', 'election', '2024-03-01', 10, "election"))
    # print(db.get_flower_post('twitter', 'election', 'election', '2024-02-01', 10))
    # print(db.get_flower_post_link('election', 'election', '2024-02-01', 10, '2024-02-01', '2024-02-10'))
    # print(db.get_flower_layout(['weibo', 'twitter','facebook'], 'election','2024-02-01', 10))
    # print(db.get_main_view('election', 'election', '2024-02-01', 10,))
    # {
    #     "twitter":['1749159384112845285','1765396693036523717'],
    #     'weibo':['1749131121114091648','1386635238986510341']
    # }))
    # print(db.get_history_relevance('election', 'election', '2024-02-01', 10,'1749159384112845285','1749131121114091648'))
    # print(db.get_interest_distribution(['weibo', 'twitter'], 'election', '2024-04-01', 10000, '2467791', '2836421'))
    #
