from fastapi import APIRouter
from app.api.v1 import predict

router = APIRouter()
router.include_router(predict.router)
