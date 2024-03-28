from fastapi import APIRouter
from typing import List, Dict, Union
from pydantic import BaseModel
from utils import database
import os

cp = APIRouter()
# print(os.getcwd())
db = database.DataBase('./data/case1/all_posts.csv','./data/case1/all_accounts.csv')

class TimeLineRequest(BaseModel):
    names: List[str]
    event: str


class TopicRequest(BaseModel):
    names: List[str]
    event: str
    date: str
    cycle: int


class FlowerRequest(BaseModel):
    names: Union[str, List[str]] = None
    event: str
    date: str
    cycle: int
    # topic为可省略
    topic: str = None
    start: str = None
    end: str = None


@cp.post("/time_line/")
async def get_post_count(request: TimeLineRequest):
    response_data = {}
    event = request.event
    names = request.names
    for name in names:
        # 返回所有日期以及发帖数
        response_data[name] = db.get_time_line(name, event)
    print(response_data)
    return response_data


@cp.post("/topic/")
async def get_topic(request: TopicRequest):
    # 获取需要获得的事件名称
    event = request.event
    # 获取需要获得的日期
    date = request.date
    # 获取向前计算的周期
    cycle = request.cycle
    # 获取所有的平台名称
    names = request.names
    # 获取聚类
    response_data = db.get_topic(names, event, date, cycle)
    return response_data


@cp.post("/flower/")
async def get_flower(request: FlowerRequest):
    platform_names = request.names
    event = request.event
    date = request.date
    cycle = request.cycle
    return db.get_flower(platform_names, event, date, cycle)


class FlowerWordCloudRequest(BaseModel):
    names: str
    event: str
    date: str
    cycle: int
    topic: str

@cp.post("/flower_word_cloud/")
async def get_flower_word_cloud(request: FlowerWordCloudRequest):
    platform_name = request.names
    event = request.event
    date = request.date
    cycle = request.cycle
    topic = request.topic
    return db.get_flower_word_cloud(platform_name, event, date, cycle, topic)


@cp.post("/flower_post/")
async def get_flower_post(request: FlowerWordCloudRequest):
    name = request.names
    event = request.event
    date = request.date
    cycle = request.cycle
    topic = request.topic
    output = db.get_flower_post(name, event, topic, date, cycle)
    return output


@cp.post("/flower_post_link/")
async def get_flower_post_link(request: FlowerRequest):
    event = request.event
    topic = request.topic
    date = request.date
    cycle = request.cycle
    start = request.start
    end = request.end
    return db.get_flower_post_link(event, topic, date, cycle, start, end)


@cp.post("/flower_layout/")
async def get_flower_layout(request: FlowerRequest):
    platform_names = request.names
    event = request.event
    date = request.date
    cycle = request.cycle
    return db.get_flower_layout(platform_names, event,date, cycle)


class MainViewRequest(BaseModel):
    event: str
    topic: str
    date: str
    cycle: int
    flower_post: Dict[str, List[str]]


@cp.post("/main_view/")
async def get_main_view(request: MainViewRequest):
    event = request.event
    topic = request.topic
    date = request.date
    cycle = request.cycle
    flower_post = request.flower_post
    return db.get_main_view(event, topic, date, cycle, flower_post)


class DetailViewRequest(BaseModel):
    names: List[str]
    event: str
    date: str
    cycle: int
    source: Dict[str, str]
    target: Dict[str, str]


@cp.post("/history_relevance/")
async def get_history_relevance(request: DetailViewRequest):
    names = request.names
    event = request.event
    date = request.date
    cycle = request.cycle
    source = request.source
    target = request.target
    source_keys = list(source.values())[0]
    target_keys = list(target.values())[0]
    return db.get_history_relevance(names, event, date, cycle, source_keys, target_keys)


@cp.post("/interest_distribution/")
async def get_history_relevance_post(request: DetailViewRequest):
    names = request.names
    event = request.event
    date = request.date
    cycle = request.cycle
    source = request.source
    target = request.target
    # source_keys = list(source.values())[0]
    # target_keys = list(target.values())[0]
    return db.get_interest_distribution(names, event, date, cycle, source,  target)
