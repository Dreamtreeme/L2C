from collections import OrderedDict
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingLRUCache:
    """
    쿼리 문자열에 대응하는 768차원 임베딩 벡터를 메모리 상에 보관하는 경량 LRU 캐시입니다.
    """
    def __init__(self, max_size: int = 128):
        self.cache = OrderedDict()
        self.max_size = max_size
        logger.debug(f"EmbeddingLRUCache initialized with max_size={max_size}")

    def get(self, query: str) -> List[float] | None:
        if query in self.cache:
            logger.debug(f"EmbeddingLRUCache HIT for query: {query!r}")
            self.cache.move_to_end(query)
            return self.cache[query]
        logger.debug(f"EmbeddingLRUCache MISS for query: {query!r}")
        return None

    def set(self, query: str, embedding: List[float]) -> None:
        if query in self.cache:
            self.cache.move_to_end(query)
        self.cache[query] = embedding
        if len(self.cache) > self.max_size:
            popped_query, _ = self.cache.popitem(last=False)
            logger.debug(f"EmbeddingLRUCache Evicted oldest query: {popped_query!r}")
