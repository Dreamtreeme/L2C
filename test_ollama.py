import ollama
from shared.config import EXTRACTION_PROMPT
import json

text = "이곳은 토스입니다. Backend Engineer를 모십니다. Java, Kotlin을 사용합니다. 연봉은 1억입니다. 학력은 무관입니다. 서울 강남구에서 근무합니다."
prompt = EXTRACTION_PROMPT.format(text=text)

res = ollama.chat(
    model='qwen3:4b',
    messages=[{'role': 'user', 'content': prompt}],
    format='json'
)

with open("test_ollama.txt", "w", encoding="utf-8") as f:
    f.write(res['message']['content'])
