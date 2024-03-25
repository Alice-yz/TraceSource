import requests

# 请求的 URL
url = 'http://127.0.0.1:8000/cp/topic/'

# 请求头部信息
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

# 请求体数据
data = {
    "names": [
        "twitter"
    ],
    "event": "election",
    "date": "2024-04-01",
    "cycle": 17
}

# 发送 POST 请求
response = requests.post(url, headers=headers, json=data)

# 打印响应结果
print(response.text)