import json
import base64
import requests
from PIL import Image, ImageDraw, ImageFont

def test_ollama():
    image_path = r"C:\Users\psg\Desktop\L2C\data\screenshots\screen_20260520_073042.png"
    
    img = Image.open(image_path)
    width, height = img.size
    print(f"Original Resolution: {width}x{height}")
    
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
        
    prompt = """
Analyze this UI screenshot of a Korean website. Extract at most 8 most important clickable elements (buttons, inputs, tabs, search/login buttons).
당신은 반드시 단일 JSON 객체(key: "elements") 형태로만 응답해야 합니다. 마크다운 코드블록 기호(```)나 설명 텍스트를 절대 쓰지 말고, 오직 { "elements": [ ... ] } 형태의 JSON 자체만을 출력하세요.

Each object in the "elements" array must have:
- "id": integer (start from 0)
- "text": visible text or description of the icon (e.g. "검색창", "돋보기", "탐색")
- "bbox": [xmin, ymin, xmax, ymax] representing the bounding box, normalized to 0-1000 scale.

Example output format (MUST follow this structure):
{
  "elements": [
    {"id": 0, "text": "검색창", "bbox": [350, 50, 450, 100]}
  ]
}
"""

    # Test 1: WITHOUT JSON format
    print("\n--- Test 1: Without format='json' ---")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5vl:3b",
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "num_ctx": 2048,
                    "num_predict": 512,
                    "temperature": 0.1
                }
            },
            timeout=120
        )
        response.raise_for_status()
        resp_json = response.json()
        resp_text = resp_json.get("response", "").strip()
        print("Response Text:\n", resp_text)
    except Exception as e:
        print("Test 1 Failed:", e)

    # Test 2: WITH JSON format but simplified prompt
    print("\n--- Test 2: With format='json' and simplified prompt ---")
    simple_prompt = """Extract clickable elements in this screenshot as JSON.
Format:
{
  "elements": [
    {"id": 0, "text": "description", "bbox": [xmin, ymin, xmax, ymax]}
  ]
}
Note: bbox should be normalized to 0-1000 scale. Keep it concise.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5vl:3b",
                "prompt": simple_prompt,
                "images": [base64_image],
                "stream": False,
                "format": "json",
                "options": {
                    "num_ctx": 2048,
                    "num_predict": 512,
                    "temperature": 0.1
                }
            },
            timeout=120
        )
        response.raise_for_status()
        resp_json = response.json()
        resp_text = resp_json.get("response", "").strip()
        print("Response Text:\n", resp_text)
    except Exception as e:
        print("Test 2 Failed:", e)

if __name__ == "__main__":
    test_ollama()
