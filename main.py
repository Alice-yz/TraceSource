import uvicorn
from fastapi import FastAPI

from apps import *
# import os
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
app = FastAPI()
app.include_router(cp, prefix="/cp")

if __name__ == "__main__":
    uvicorn.run('main:app',
                port=8000,
                reload=True,
                reload_delay=0.1)
