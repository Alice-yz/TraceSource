import json
import pandas as pd
from csv_tools import read_csv_post, read_csv_user

case_posts = json.load(open('./data/post_highlight.json', 'r', encoding='utf-8'))
case_posts_other = json.load(open('./data/post_others.json', 'r', encoding='utf-8'))

encountered_post_id = {"weibo": [], "twitter": [], "facebook":[]}

all_posts = read_csv_post('./data/all_posts.csv')

filtered_posts = []

post_columns=['from', 'post_id','text_trans','text','publish_time','cnt_retweet','cnt_agree','cnt_comment','user_id','screen_name','name','url','action','avatar','img','query','sentiment_score','cluster','event']
df_filtered_posts = pd.DataFrame([], columns=post_columns)

cluster_list = [
    'Great_Wave_Kanagawa',
    '240_china_nuclear_pollution',
    'cooling_water_nuclear_wastewater',
    '70_billion_japan_water',
    'sue_TEPCO_japan',
    'south_korea_nuclear_discharge',
    'japan_dead_fish',
    'border_united_texas_trump'
]

for topic in case_posts:
    if topic not in cluster_list:
        continue
    for post_pair in case_posts[topic]:
        src_post = post_pair['source']
        tgt_post = post_pair['target']
        if src_post['id'] not in encountered_post_id[src_post['from']]:
            encountered_post_id[src_post['from']].append(src_post['id'])
            rows_to_add = all_posts[(all_posts['post_id'] == src_post['id']) & (all_posts['cluster'] == topic)]
            df_filtered_posts = pd.concat([df_filtered_posts, rows_to_add], ignore_index=True)
        if tgt_post['id'] not in encountered_post_id[tgt_post['from']]:
            encountered_post_id[tgt_post['from']].append(tgt_post['id'])
            rows_to_add = all_posts[(all_posts['post_id'] == tgt_post['id']) & (all_posts['cluster'] == topic)]
            df_filtered_posts = pd.concat([df_filtered_posts, rows_to_add], ignore_index=True)
"""
for topic in case_posts_other:
    if topic not in cluster_list:
        continue
    for post_pair in case_posts_other[topic]:
        src_post = post_pair['source']
        tgt_post = post_pair['target']
        if src_post['id'] not in encountered_post_id[src_post['from']]:
            encountered_post_id[src_post['from']].append(src_post['id'])
            rows_to_add = all_posts[(all_posts['post_id'] == src_post['id']) & (all_posts['cluster'] == topic)]
            df_filtered_posts = pd.concat([df_filtered_posts, rows_to_add], ignore_index=True)
        if tgt_post['id'] not in encountered_post_id[tgt_post['from']]:
            encountered_post_id[tgt_post['from']].append(tgt_post['id'])
            rows_to_add = all_posts[(all_posts['post_id'] == tgt_post['id']) & (all_posts['cluster'] == topic)]
            df_filtered_posts = pd.concat([df_filtered_posts, rows_to_add], ignore_index=True)
"""
df_filtered_posts.sort_values(by=['cluster', 'from'], ascending=False, inplace=True)

# 将df_filtered_posts存入filtered_posts.csv中
df_filtered_posts.to_csv('./data/filtered_posts.csv', encoding='utf-8', index=False)
