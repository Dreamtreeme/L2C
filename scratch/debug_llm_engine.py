import os
import sys
import traceback
import dotenv

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from classic.extractor.llm_engine import LLMEngine

sample_text = """
회사명: 글로벌머니익스프레스
직무: Mobile 개발자(iOS) (3년 이상)
주요업무:
• 글로벌 송금 앱의 Android 클라이언트 개발 및 유지보수
• 앱 성능 개선 및 버그 수정
자격요건:
• iOS 앱 개발 경력 3년 이상 ~ 15년 이하
• Swift 및 UIKit에 대한 깊은 이해
"""

try:
    engine = LLMEngine()
    res = engine.extract_from_text(sample_text)
    print("Success:", res)
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
