from collections import deque
import json

twitter_hex_directions = {
    "E": (1, -1, 0),
    "SE": (0, -1, 1),
    "SW": (-1, 0, 1),
    "W": (-1, 1, 0),
    "NW": (0, 1, -1),
    "NE": (1, 0, -1)
}

weibo_hex_directions = {
    "E": (1, -1, 0),
    "NE": (1, 0, -1),
    "NW": (0, 1, -1),
    "W": (-1, 1, 0),
    "SW": (-1, 0, 1),
    "SE": (0, -1, 1)
}

facebook_hex_directions = {
    "SW": (-1, 0, 1),
    "SE": (0, -1, 1),
    "E": (1, -1, 0),
    "NE": (1, 0, -1),
    "NW": (0, 1, -1),
    "W": (-1, 1, 0)
}

# 六边形坐标->六边形端点坐标的转换
vertices = {
    'N': (1,1,0),
    'NE': (1,0,0),
    'SE': (1,0,1),
    'S': (0,0,1),
    'SW': (0,1,1),
    'NW': (0,1,0)
}

# 人的中心坐标->人的端点坐标的转换
ren_directions = {
    'N':(0,0,-1),
    'SE':(0,-1,0),
    'SW':(-1,0,0)
}
# 丫的中心坐标->丫的端点坐标的转换
ya_directions = {
    'NE':(1,0,0),
    'S':(0,0,1),
    'NW':(0,1,0)
}

def identify_struct(dir):
    if dir in ren_directions.keys():
        return "ren"
    elif dir in ya_directions.keys():
        return "ya"
def reversal_struct(struct):
    if struct == "ren":
        return "ya"
    elif struct == "ya":
        return "ren"

def get_vertices(hex):
    """计算一个六边形的所有顶点坐标"""
    x, y, z = hex
    return {dir: (x + dx, y + dy, z + dz) for dir, (dx, dy, dz) in vertices.items()}

def get_hex_neighbors(hex, platform_name):
    x, y, z = hex
    if platform_name == "twitter":
        return [(x + dx, y + dy, z + dz) for dir, (dx, dy, dz) in twitter_hex_directions.items()]
    if platform_name == "weibo":
        return [(x + dx, y + dy, z + dz) for dir, (dx, dy, dz) in weibo_hex_directions.items()]
    if platform_name == "facebook":
        return [(x + dx, y + dy, z + dz) for dir, (dx, dy, dz) in facebook_hex_directions.items()]

def expand_cluster(platform_name, center, count):
    cluster = [center]
    queue = deque([center])

    while queue and len(cluster) < count:
        current = queue.popleft()
        for neighbor in get_hex_neighbors(current, platform_name):
            if neighbor not in cluster:
                cluster.append(neighbor)
                queue.append(neighbor)
                if len(cluster) == count:
                    break

    return cluster

# 立方体坐标系->平面直角坐标系 (原点重合，边朝上的六边形)
def hex_to_pixel_2(hex):
    a, b, c = hex
    return -0.5*a-0.5*b+c, 3**0.5/2*a - 3**0.5/2*b

def generate_cluster_level_layout(platform_name_list, topic_prop_ability):
    """
    [函数输入参数说明]
    platform_name_list: list, e.g. ["twitter", "weibo", "facebook"]
    topic_prop_ability: dict, 
                        key: topic_name, 
                        value: list[source_platform_name, target_platform_name, cross_platform_propagation_ability]
        e.g.
        topic_prop_ability = {
            "donald_trump": [["twitter", "weibo", 0.7], ["twitter", "facebook", 0.4], ["facebook", "weibo", 0.5]],
            "john_bidon": [["twitter", "weibo", 0.6], ["facebook", "twitter", 0.2], ["facebook", "weibo", 0.5]],
            ...
        }
    [函数返回值说明]
    我会遵循API需求文档中Cluster Level下对[GET]flower-layout的返回结果要求, 来返回结果
    """
    topic_num = len(topic_prop_ability)
    topic_list = list(topic_prop_ability.keys())
    platform_num = len(platform_name_list)
    layout_results = {
        "clusters":{},
        "paths":[]
    }
    for i in range(platform_num):
        platform = platform_name_list[i]
        layout_results["clusters"][platform] = []

    # 1. 生成每个平台的glyph聚落的中心六边形坐标
    platform_clusters_centers = {"twitter":(0,0,0), "weibo":(0,0,0), "facebook":(0,0,0)}
    # if(topic_num <= 7):
    #     platform_clusters_centers["twitter"] = (1,1,-2)
    #     platform_clusters_centers["weibo"] = (-1,-1,2)
    #     platform_clusters_centers["facebook"] = (-3,3,0)
    # elif(topic_num <= 19):
    #     platform_clusters_centers["twitter"] = (2,2,-4)
    #     platform_clusters_centers["weibo"] = (-2,-2,4)
    #     platform_clusters_centers["facebook"] = (-4,4,0)
    # elif(topic_num <= 37):
    #     platform_clusters_centers["twitter"] = (3,3,-6)
    #     platform_clusters_centers["weibo"] = (-3,-3,6)
    #     platform_clusters_centers["facebook"] = (-5,5,0)
    
    accumulate_hex_number = 1
    for i in range(topic_num):
        accumulate_hex_number = accumulate_hex_number + (i+1) * 6
        if(topic_num <= accumulate_hex_number):
            platform_clusters_centers["twitter"] = (i+1,i+1,-2*(i+1))
            platform_clusters_centers["weibo"] = (-(i+1),-(i+1),2*(i+1))
            platform_clusters_centers["facebook"] = (-i-3, i+3, 0)
            break
    
    # 2. 生成每个glyph中心的坐标
    platform_glyphs_centers = {}
    for platform in platform_name_list:
        platform_glyphs_centers[platform] = expand_cluster(platform, platform_clusters_centers[platform], topic_num)
        pixel_coords = [hex_to_pixel_2(coor) for coor in platform_glyphs_centers[platform]]
        for i in range(topic_num):
            layout_results["clusters"][platform].append({
                "topic_name": topic_list[i],
                "x": pixel_coords[i][0],
                "y": pixel_coords[i][1]
            })
        
    # 3. 生成同话题glyph之间的传播路径
    # 全局字典来存储经过的点的激励值
    vertex_incentives = {}
    def get_neighbors(current_vertex):
        """计算一个六边形顶点的所有邻居顶点以及到达邻居的边，激励值高的放前面"""
        x, y, z = current_vertex[0]
        if current_vertex[1] == "ren":
            directions_dict = ren_directions
        elif current_vertex[1] == "ya":
            directions_dict = ya_directions

        neighbors = {}
        for dir, (dx, dy, dz) in directions_dict.items():
            neighbor = (x + dx, y + dy, z + dz)

            # 为每个顶点设置初始激励值
            if neighbor not in vertex_incentives:
                vertex_incentives[neighbor] = 0

            neighbors[dir] = neighbor

        # 根据激励值对邻居进行排序，激励值高的优先
        sorted_neighbors = {dir: neighbors[dir] for dir in sorted(neighbors, key=lambda d: vertex_incentives[neighbors[d]], reverse=True)}

        return sorted_neighbors

    def bfs(start_info, end_vertices):
        """广度优先搜索算法，记录经过的边"""
        queue = deque([start_info]) # [[start坐标, "ren"or"ya"]]
        visited = {start_info[0]: None}
        edges = {}  # 保存到达每个节点的相对于六边形中心的位置
        
        while queue:
            current_vertex = queue.popleft()
            # print(current_vertex)
            if current_vertex[0] in end_vertices:
                break
            for dir, neighbor in get_neighbors(current_vertex).items(): # get_neighbors函数需要重写，人的中心走一步一定是丫，丫的中心走一步一定是人
                if neighbor not in visited:
                    queue.append([neighbor, reversal_struct(current_vertex[1])])
                    visited[neighbor] = current_vertex # 记录是从哪个顶点到达这个顶点坐标的
                    edges[neighbor] = dir  # 记录相对于中心的位置
        
        # 回溯找路径中经过的边
        path_vertices = []
        while current_vertex[0] != start_info[0]:  # 回溯直到起点
            path_vertices.append(current_vertex[0])
            prev_vertex = visited[current_vertex[0]]

            # 为经过的点更新激励值
            vertex_incentives[current_vertex[0]] = vertex_incentives.get(current_vertex[0], 0) + 1

            current_vertex = prev_vertex
        path_vertices.append(start_info[0])
        
        return path_vertices[::-1]

    for i, t in enumerate(topic_prop_ability):
        for prop_res in topic_prop_ability[t]:
            source_platform = prop_res[0]
            target_platform = prop_res[1]
            prop_ability = prop_res[2]

            start = platform_glyphs_centers[source_platform][i]
            end = platform_glyphs_centers[target_platform][i]
            # 到达终点六边形的一个端点就算到达终点
            end_vertices = [coor for dir, coor in get_vertices(end).items()]
            # real_start选择距离end最近的那个
            start_vertices = [[coor, identify_struct(dir)] for dir, coor in get_vertices(start).items()]
            min_distance = (start_vertices[0][0][0]-end[0])**2 + (start_vertices[0][0][1]-end[1])**2 + (start_vertices[0][0][2]-end[2])**2
            real_start = start_vertices[0][0]
            real_start_struct = start_vertices[0][1]
            for start_vertex in start_vertices:
                distance = (start_vertex[0][0]-end[0])**2 + (start_vertex[0][1]-end[1])**2 + (start_vertex[0][2]-end[2])**2
                if distance < min_distance:
                    min_distance = distance
                    real_start = start_vertex[0]
                    real_start_struct = start_vertex[1]
            real_start_info = [real_start, real_start_struct]

            # 执行BFS搜索
            # print("Real Start:", real_start_info)
            # print("Real End Candidates:", end_vertices)
            path_vertices = bfs(real_start_info, end_vertices)
            print("Vertices:", path_vertices)
            print('Start Hex(Rectangular Coordinates):', hex_to_pixel_2(start))
            print('End Hex(Rectangular Coordinates):', hex_to_pixel_2(end))
            path_vertices.insert(0,start)
            path_vertices.append(end)
            print("Vertices(Rectangular Coordinates):", [hex_to_pixel_2(path_vertex) for path_vertex in path_vertices])
            print("----------------------------------------------------------------------------------------------")
            pixel_path = [hex_to_pixel_2(path_vertex) for path_vertex in path_vertices]
            pixel_path_points = []
            for k in range(len(pixel_path)):
                pixel_path_points.append({
                    "x": pixel_path[k][0],
                    "y": pixel_path[k][1]
                })
            layout_results["paths"].append({
                "start": source_platform,
                "end": target_platform,
                "topic": t,
                "weight": prop_ability,
                "points": pixel_path_points
            })

    return layout_results


if "__name__" == "__main__":
    # # 测试代码
    platform_name_list = ["twitter", "weibo", "facebook"]
    topic_prop_ability = {
        "donald_trump": [["twitter", "weibo", 0.7], ["twitter", "facebook", 0.4], ["facebook", "weibo", 0.5]],
        "john_bidon": [["weibo", "twitter", 0.5], ["facebook", "twitter", 0.3], ["facebook", "weibo", 0.6]],
        "hillary_clinton": [["weibo", "twitter", 0.6], ["facebook", "weibo", 0.7], ["facebook", "twitter", 0.4]],
        "barack_obama": [["twitter", "weibo", 0.8], ["facebook", "weibo", 0.6], ["facebook", "twitter", 0.3]],
        "joe_biden": [["twitter", "weibo", 0.9], ["facebook", "weibo", 0.7], ["facebook", "twitter", 0.5]],
        "mike_pence": [["weibo", "twitter", 0.4], ["facebook", "weibo", 0.6], ["facebook", "twitter", 0.3]],
        "kamala_harris": [["weibo", "twitter", 0.3], ["facebook", "weibo", 0.5], ["facebook", "twitter", 0.2]]
    }
    layout_results = generate_cluster_level_layout(platform_name_list, topic_prop_ability)
    # print(layout_results)
    # 使用json.dumps方法美化打印
    pretty_print = json.dumps(layout_results, indent=4)
    print(pretty_print)