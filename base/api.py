import os
import logging
from typing import Union
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException, Request, APIRouter
from fastapi.responses import PlainTextResponse
from starlette.status import HTTP_401_UNAUTHORIZED
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast
from sklearn.preprocessing import PolynomialFeatures

# 环境变量
# cpu或cuda 用于在Dockerfile中传入 这将传递给SentenceTransformer
DEVICE = os.environ.get("DEVICE", "cuda")
API_KEY = os.environ.get("API_KEY", "")
# MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3-unsupervised")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", os.path.join(os.getcwd(), "model"))
TOKENIZER_FILE = os.environ.get(
    "TOKENIZER_FILE", os.path.join(os.getcwd(), "model", "tokenizer.json")
)
# CUDA_LAUNCH_BLOCKING = 1

logger = logging.getLogger(__name__)

router = APIRouter()

# 计算token数量的模型
tokenizer_model_instance: PreTrainedTokenizerFast = None

model_instance: SentenceTransformer = None


def get_model_instance(reload: bool = False) -> SentenceTransformer:
    """
    获取可用模型
    """
    global model_instance
    if model_instance is None:
        logger.info(f"读取模型: {MODEL_NAME}, device: {DEVICE}")
        # device="cuda", "cpu"
        # compute_type=(GPU with FP16 "float16"), (GPU with INT8 "int8_float16"), (CPU with INT8 "int8")
        model_instance = SentenceTransformer(MODEL_NAME, device=DEVICE)
    if reload:
        logger.info(f"重新读取模型: {MODEL_NAME}, device: {DEVICE}")
        model_instance = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return model_instance


def get_tokenizer_model_instance(reload: bool = False) -> PreTrainedTokenizerFast:
    """
    获取token计算模型
    """
    global tokenizer_model_instance
    if tokenizer_model_instance is None:
        logger.info(f"读取分词器")
        tokenizer_model_instance = PreTrainedTokenizerFast(
            tokenizer_file=TOKENIZER_FILE
        )
    if reload:
        logger.info(f"读取分词器")
        tokenizer_model_instance = PreTrainedTokenizerFast(
            tokenizer_file=TOKENIZER_FILE
        )
    return tokenizer_model_instance


get_model_instance()
get_tokenizer_model_instance()


class EmbeddingProcessRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingQuestionRequest(BaseModel):
    input: str
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


async def verify_token(request: Request):
    if API_KEY == "":
        return True
    auth_header = request.headers.get("Authorization")
    if auth_header:
        token_type, _, token = auth_header.partition(" ")
        if token_type.lower() == "bearer" and token == API_KEY:
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(get_tokenizer_model_instance().encode(string))
    return num_tokens


def expand_features(embedding, target_length):
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
        expanded_embedding = np.pad(
            expanded_embedding, (0, target_length - len(expanded_embedding))
        )
    return expanded_embedding


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: Union[EmbeddingProcessRequest, EmbeddingQuestionRequest]
):
    """
    `/v1/embeddings` 接口的模拟实现

    参数详细说明参见 OpenAI 文档: https://platform.openai.com/docs/api-reference/embeddings/create
    """
    await verify_token(request)

    if isinstance(request, EmbeddingProcessRequest):
        logger.debug("EmbeddingProcessRequest")
        payload = request.input
    elif isinstance(request, EmbeddingQuestionRequest):
        logger.debug("EmbeddingQuestionRequest")
        payload = [request.input]
    else:
        logger.debug("Request")
        data = request.json()
        logger.debug(data)
        return

    logger.debug(payload)

    embeddings = []
    try:
        # 计算嵌入向量和tokens数量
        embeddings = [get_model_instance().encode(text) for text in payload]
        logger.info(f"embeddings: {len(embeddings)}")
    except Exception as e:
        logger.error(f"模型处理过程中出现错误: {e}")
        # 重新读取模型
        get_model_instance(reload=True)
        return HTTPException(detail="模型处理过程中出现错误")

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
    # embeddings = [interpolate_vector(embedding, 1536) if len(embedding) < 1536 else embedding for embedding in embeddings]
    # 如果嵌入向量的维度不为1536，则使用特征扩展法扩展至1536维度
    # embeddings = [
    #     expand_features(embedding, 1536) if len(embedding) < 1536 else embedding
    #     for embedding in embeddings
    # ]

    # Min-Max normalization
    # embeddings = [(embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)) if np.max(embedding) != np.min(embedding) else embedding for embedding in embeddings]
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in payload)
    total_tokens = sum(num_tokens_from_string(text) for text in payload)

    response = {
        "object": "list",
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }

    return response


@router.post("/v1/chat/completions")
async def chat_completions():
    """
    `/v1/chat/completions` 接口的模拟实现, 用于oneapi的ping测试, 返回状态码 200 和空字符串
    """
    return PlainTextResponse(content="", status_code=200)
