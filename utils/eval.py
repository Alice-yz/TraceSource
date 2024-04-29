import json

def find_dicts_intersection(list1, list2):
    # 使用集合存储交集结果，避免重复
    intersection = []
    for d1 in list1:
        for d2 in list2:
            if d1 == d2:
                intersection.append(d1)
                break
    return intersection

def to_frozenset(d):
    """递归地将字典转换为frozenset，以便可以进行哈希和比较。"""
    if isinstance(d, dict):
        return frozenset((k, to_frozenset(v)) for k, v in d.items())
    elif isinstance(d, list):
        return tuple(to_frozenset(item) for item in d)
    return d

def find_dicts_union(list1, list2):
    # 创建一个集合存储不可变的字典表示
    seen = set()
    union_list = []

    for item in list1 + list2:
        # 将每个字典转换为不可变形式
        fs_item = to_frozenset(item)
        # 如果不在已见集合中，则添加到结果列表
        if fs_item not in seen:
            seen.add(fs_item)
            union_list.append(item)
    
    return union_list

# cross-platform 
gt_cp = json.load(open('./data/gt_post_pair_cp.json', 'r', encoding='utf-8'))
pred_cp = json.load(open('./data/post_highlight_above_threshold.json', 'r', encoding='utf-8'))
# pred_cp2 = json.load(open('./data/post_others_above_threshold.json', 'r', encoding='utf-8'))

# considering path
for topic in gt_cp:
    gt_pairs_list = gt_cp[topic]
    pred_pairs_list = pred_cp[topic] if topic in pred_cp else []
    intersection_list = find_dicts_intersection(gt_pairs_list, pred_pairs_list)
    union_list = find_dicts_union(gt_pairs_list, pred_pairs_list)
    edge_jaccard = len(intersection_list) / len(union_list)
    print(f'{topic} edge_jaccard: {edge_jaccard}')

# inside-platform
gt_ip = json.load(open('./data/gt_post_pair_ip.json', 'r', encoding='utf-8'))
pred_ip = json.load(open('./data/post_ip_above_threshold.json', 'r', encoding='utf-8'))

# considering path
for topic in pred_ip:
    for platform in pred_ip[topic]:
        gt_pairs_list = gt_ip[topic][platform] if topic in gt_ip and platform in gt_ip[topic] else []
        pred_pairs_list = pred_ip[topic][platform]
        intersection_list = find_dicts_intersection(gt_pairs_list, pred_pairs_list)
        union_list = find_dicts_union(gt_pairs_list, pred_pairs_list)
        edge_jaccard = len(intersection_list) / len(union_list) if len(union_list) > 0 else None
        print(f'topic: {topic}\tplatform: {platform}\tedge_jaccard: {edge_jaccard}')