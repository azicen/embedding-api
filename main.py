import logging
import os
from api import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROUTE_PREFIX = os.environ.get('ROUTE_PREFIX', '')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(route_prefix=ROUTE_PREFIX)

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=38000)
