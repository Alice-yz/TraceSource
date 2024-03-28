import uvicorn
from fastapi import FastAPI

from apps import *
from fastapi.middleware.cors import CORSMiddleware
# import os
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
app = FastAPI()
app.include_router(cp, prefix="/cp")

# 允许跨域请求的来源
origins = [
    "*",
]

# 设置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run('main:app',
                port=8000,
                reload=True,
                reload_delay=0.1)
