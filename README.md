# 1 安装
版本都在这里如果有啥报错可以看看是不是一个版本
```text
bleach                    6.1.0
charset-normalizer        3.3.2
click                     8.1.7
colorama                  0.4.6
comm                      0.2.1
contourpy                 1.2.0
cycler                    0.12.1
Cython                    0.29.37
debugpy                   1.8.1
decorator                 5.1.1
defusedxml                0.7.1
exceptiongroup            1.2.0
executing                 2.0.1
fastapi                   0.110.0
fastjsonschema            2.19.1
filelock                  3.13.1
fonttools                 4.49.0
fqdn                      1.5.1
fsspec                    2024.2.0
funcy                     2.0
gensim                    4.3.2
h11                       0.14.0
hdbscan                   0.8.33
httpcore                  1.0.4
httptools                 0.6.1
httpx                     0.27.0
huggingface-hub           0.20.3
idna                      3.6
importlib-metadata        7.0.1
importlib-resources       6.1.1
ipykernel                 6.29.2
ipython                   8.18.1
isoduration               20.11.0
jedi                      0.19.1
jellyfish                 1.0.3
jieba                     0.42.1
Jinja2                    3.1.3
joblib                    1.3.2
json5                     0.9.17
jsonpointer               2.4
jsonschema                4.21.1
jsonschema-specifications 2023.12.1
jupyter_client            8.6.0
jupyter_core              5.7.1
jupyter-events            0.9.0
jupyter-lsp               2.2.2
jupyter_server            2.12.5
jupyter_server_terminals  0.5.2
jupyterlab                4.1.2
jupyterlab_pygments       0.3.0
jupyterlab_server         2.25.3
kiwisolver                1.4.5
langdetect                1.0.9
Levenshtein               0.25.0
llvmlite                  0.42.0
MarkupSafe                2.1.5
matplotlib                3.8.3
matplotlib-inline         0.1.6
mistune                   3.0.2
mpmath                    1.3.0
nbclient                  0.9.0
nbconvert                 7.16.1
nbformat                  5.9.2
nest-asyncio              1.6.0
networkx                  3.2.1
nltk                      3.8.1
notebook_shim             0.2.4
numba                     0.59.0
numexpr                   2.9.0
numpy                     1.26.4
overrides                 7.7.0
packaging                 23.2
pandas                    2.2.0
pandocfilters             1.5.1
parso                     0.8.3
pillow                    10.2.0
pip                       24.0
platformdirs              4.2.0
plotly                    5.19.0
prometheus_client         0.20.0
prompt-toolkit            3.0.43
psutil                    5.9.8
pure-eval                 0.2.2
pycparser                 2.21
pydantic                  2.6.4
pydantic_core             2.16.3
Pygments                  2.17.2
pyLDAvis                  3.4.1
pynndescent               0.5.11
pyparsing                 3.1.1
python-dateutil           2.8.2
python-dotenv             1.0.1
python-json-logger        2.0.7
pytz                      2024.1
pywin32                   306
pywinpty                  2.0.12
PyYAML                    6.0.1
pyzmq                     25.1.2
rapidfuzz                 3.6.2
referencing               0.33.0
regex                     2023.12.25
requests                  2.31.0
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rpds-py                   0.18.0
safetensors               0.4.2
scikit-learn              1.4.1.post1
scipy                     1.12.0
Send2Trash                1.8.2
sentence-transformers     2.3.1
sentencepiece             0.2.0
setuptools                69.1.0
six                       1.16.0
smart-open                7.0.1
sniffio                   1.3.0
soupsieve                 2.5
stack-data                0.6.3
starlette                 0.36.3
sympy                     1.12
tenacity                  8.2.3
terminado                 0.18.0
threadpoolctl             3.3.0
tinycss2                  1.2.1
tokenizers                0.15.2
tomli                     2.0.1
torch                     2.2.1
tornado                   6.4
tqdm                      4.66.2
uvicorn                   0.28.0
```
# 2 运行
入口在main.py，可以使用test_json.jsonl，里面的做请求测试
# 3 项目结构
```text
app 
    ├── cross_platform.py # 后端api
data
    ├── all_post.csv # 所有的post数据 在cross_platform.py中被加载
    ├── all_account.csv # 所有的account数据 在cross_platform.py中被加载
    ├── cluster.json # 所有计算好的cluster之间的传播分数 在database.py中被加载
    ├── post_highlight.json # 所有计算好一定会被显示传播关系的post(及就是所有PPT上指定具体信息的post) 在database.py中被加载
    ├── post_other.json # 其它可能会被显示出传播关系的post 在database.py中被加载
utils
    ├── database.py # 加载数据，并且包含后端逻辑
    ├── cal_cluster_factor.py # 计算cluster之间的传播分数
    ├── cal_post_factor.py # 计算post之间的传播分数
    ├── cluster_level_main_view_layout.py # 布局
    ├── freq.py # 词频统计中的函数
    ├── lda.py # lda聚类中的函数
main.py # 入口
test_json.jsonl # 测试数据，可以在127.0.0.1:8000/docs中测试
```    
# 4. 关于data中的数据构造
## 4.1 all_post.csv
|字段| 含义                                             |
|----|:-----------------------------------------------|
|from| 来源                                             |
|post_id| 帖子id                                           |
|text_trans| 翻译后的文本                                         |
|text| 文本                                             |
|publish_time| 发布时间                                           |
|cnt_retweet| 转发数                                            |
|cnt_comment| 评论数                                            |
|cnt_agree| 点赞数                                            |
|user_id| 用户id                                           |
|screen_name| 用户昵称                                           |
|name| 用户姓名                                           |
|url| 该post的链接                                       |
|action| 动作 retweet or post                             |
|avatar| 头像链接                                           |
|img| 帖子图片链接                                         |
|query| 查询关键词，注意这里只要为highlight的代表PPT中case给定具体post内容的帖子 |
|sentiment_score| 情感分数                                           |
|cluster| 聚类名称，由下划线链接                                    |
|event| 该字段被废弃                                         |
---
具体生成方式：
1. 将各种采集得到的不同平台不同格式的数据通过脚本转换为上述格式的csv
2. 对于需要翻译的文本，使用百度翻译api进行翻译
3. 对情感分数进行计算
4. 手动构造聚类，该过程几乎不可复现
---
比如拿srtp的数据举例，首先就是遍历原始的jsonl数据，把这些数据存成一个json数组
```python
# 读取数据
import os
DATA_PATH = './srtp'
all_accounts = []
all_posts = []

for filedir in os.listdir(DATA_PATH):
    count_post = 0
    for filename in os.listdir(os.path.join(DATA_PATH, filedir)):
        if filename.endswith('.jsonl'):
            # 构建完整文件路径
            file_path = os.path.join(DATA_PATH, filedir, filename)
            with open(file_path, 'r',encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if "gender" in data:
                        all_accounts.append(data)
                    else:
                        data['query'] = filedir
                        all_posts.append(data)
                        count_post += 1
    print(f'{filedir} has {count_post} posts')
```
然后就是把这些数据字段更改为上述格式，这一步就是不同的数据源的数据格式不一样，需要手动更改，需要注意`data['text'] = data['content'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(',', '，')`这一步是为了把文本中的换行符等替换掉，因为csv文件中的换行符会导致数据分行
```python
for data in all_posts:
    data['from'] = 'weibo'
    data['post_id'] = data['_id']
    data['text_trans'] = ""
    data['text'] = data['content'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(',', '，')
    data['publish_time'] = data['created_at']
    data['cnt_retweet'] = data['reposts_count']
    data['cnt_comment'] = data['comments_count']
    data['cnt_agree'] = data['attitudes_count']
    data['user_id'] = data['user']['_id'] 
    data['screen_name'] = data['user']['nick_name']
    data['name'] = data['user']['nick_name']
    data['url'] = data['url']
    if data['is_retweet']:
        data['action'] = 'retweet'
    else:
        data['action'] = 'post'
    data['avatar'] = data['user']['avatar_hd']
    if len(data['pic_urls']) > 0:
        data['img'] = data['pic_urls'][0]
    else:
        data['img'] = None
    data['query'] = data['query']
    data['sentiment_score'] = 0  # 这些先置零
    data['cluster'] = None # 这些先置空
    data['event'] = None # 废弃
```
最后就是把这些数据写入csv文件
```python
columns=['from', 'post_id','text_trans','text','publish_time','cnt_retweet','cnt_agree','cnt_comment','user_id','screen_name','name','url','action','avatar','img','query','sentiment_score','cluster','event']
df_posts = pd.DataFrame(all_posts, columns=columns)
print(df_posts.shape)
```
其它文件的处理方式都是同理的
## 4.2 all_account.csv
|字段| 含义     |
|----|:-------|
|user_id| 用户id   |
|avatar| 头像链接   |
|screen_name| 用户昵称   |
|name| 用户姓名   |
|description| 用户描述   |
|fan| 粉丝数    |
|validation| 是否验证   |
|type| 账户类型   |
|verified_reason| 验证原因   |
|screen_name_trans| 用户昵称翻译 |
|description_trans| 用户描述翻译 |
|verified_reason_trans| 验证原因翻译 |

具体生成方式：
1. 将各种采集得到的不同平台不同格式的数据通过脚本转换为上述格式的csv
2. 对于需要翻译的文本，使用百度翻译api进行翻译
## 4.3 cluster.json
```text
{
  "cluster_name": [
    [
      "platform_1",
      "platform_2",
      cluster_factor
    ]
  ],
  ...
}
```
该数据通过./data/case1/persistence_cluster.py生成
## 4.4 post_highlight.json
```text
{
  {
    "cluster_name": [
        {
            "source": {
                "id": "1385461980857716736",
                "platform": "twitter",
                "highlight": [
                    {
                        "begin": 74,
                        "end": 78
                    },
                    {
                        "begin": 194,
                        "end": 199
                    }
                ],
                "text": "........"
            },
            "target": {
                "id": "4631255079980432",
                "platform": "weibo",
                "highlight": [
                    {
                        "begin": 84,
                        "end": 91
                    }
                ],
                "text": "........"
            },
            "width": 0.518350142874509,
            "diffusion_pattern": 1,
            "is_assigned": false
        },
        ...
    ]
   },
  ...
}
```
该数据通过./data/case1/persistence_post.py生成
## 4.5 post_other.json
```text
{
  {
    "cluster_name": [
        {
            "source": {
                "id": "1385461980857716736",
                "platform": "twitter",
                "highlight": [
                    {
                        "begin": 74,
                        "end": 78
                    },
                    {
                        "begin": 194,
                        "end": 199
                    }
                ],
                "text": "........"
            },
            "target": {
                "id": "4631255079980432",
                "platform": "weibo",
                "highlight": [
                    {
                        "begin": 84,
                        "end": 91
                    }
                ],
                "text": "........"
            },
            "width": 0.518350142874509,
            "diffusion_pattern": 1,
            "is_assigned": false
        },
        ...
    ]
   },
  ...
}
```
该数据通过./data/case1/persistence_post_other.py生成