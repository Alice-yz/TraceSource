import json

TRESHOLD = 0.7
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

"""
case_posts = json.load(open('./data/post_highlight.json', 'r', encoding='utf-8'))
case_posts_other = json.load(open('./data/post_others.json', 'r', encoding='utf-8'))

result1 = {key: [] for key in case_posts} # 从case_posts中选取高于阈值的post_pair

for topic in case_posts:
    if topic not in cluster_list:
        continue
    for post_pair in case_posts[topic]:
        if post_pair['width'] >= TRESHOLD:
           result1[topic].append(post_pair)

# 将result1写入文件
with open('./data/post_highlight_above_threshold.json', 'w', encoding='utf-8') as f:
    json.dump(result1, f, ensure_ascii=False, indent=4)

result2 = {key: [] for key in case_posts_other} # 从case_posts_other中选取高于阈值的post_pair

for topic in case_posts_other:
    if topic not in cluster_list:
        continue
    for post_pair in case_posts_other[topic]:
        if post_pair['width'] >= TRESHOLD:
           result2[topic].append(post_pair)

# 将result2写入文件
with open('./data/post_others_above_threshold.json', 'w', encoding='utf-8') as f:
    json.dump(result2, f, ensure_ascii=False, indent=4)
"""

post_pairs_ip = json.load(open('./data/post_factor_ip.json', 'r', encoding='utf-8'))
result3 = {key: {'weibo':[], 'twitter':[], 'facebook':[]} for key in post_pairs_ip} # 从case_posts_ip中选取高于阈值的post_pair
for topic in post_pairs_ip:
    if topic not in cluster_list:
        continue
    for platform in post_pairs_ip[topic]:
        for post_pair in post_pairs_ip[topic][platform]:
            if post_pair['likelihood'] >= TRESHOLD and post_pair['post_rel'] >= TRESHOLD:
                result3[topic][platform].append(post_pair)
with open('./data/post_ip_above_threshold.json', 'w', encoding='utf-8') as f:
    json.dump(result3, f, ensure_ascii=False, indent=4)
