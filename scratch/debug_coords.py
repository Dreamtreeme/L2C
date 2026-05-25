import json
import base64
import requests
from PIL import Image

def debug_coords():
    image_path = r"C:\Users\psg\Desktop\L2C\data\screenshots\screen_20260520_073042.png"
    
    # 1. Print image resolution
    img = Image.open(image_path)
    width, height = img.size
    print(f"Original Screenshot Resolution: {width}x{height}")
    
    # 2. Base64 encode
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
    
    # 3. Call Ollama
    print("Calling Ollama...")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5vl:3b",
            "prompt": prompt,
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
    response.encoding = 'utf-8'
    resp_json = response.json()
    result_text = resp_json.get("response", "").strip()
    thinking_text = resp_json.get("thinking", "").strip()
    
    print("=== Raw Ollama Response ===")
    print("Response:", result_text)
    print("Thinking:", thinking_text)
    print("Full JSON:", json.dumps(resp_json, indent=2))
    print("===========================")
    
    # Use stack parser just like perception.py
    raw_markers = []
    parse_target = result_text if result_text else thinking_text
    
    if parse_target:
        try:
            parsed = json.loads(parse_target)
            if isinstance(parsed, list):
                raw_markers = parsed
            elif isinstance(parsed, dict):
                if "elements" in parsed and isinstance(parsed["elements"], list):
                    raw_markers = parsed["elements"]
        except Exception:
            pass
            
        if not raw_markers:
            i = 0
            while i < len(parse_target):
                if parse_target[i] == '{':
                    stack = 0
                    for j in range(i, len(parse_target)):
                        if parse_target[j] == '{':
                            stack += 1
                        elif parse_target[j] == '}':
                            stack -= 1
                            if stack == 0:
                                candidate = parse_target[i:j+1]
                                try:
                                    obj = json.loads(candidate)
                                    if isinstance(obj, dict) and "bbox" in obj:
                                        raw_markers.append(obj)
                                except Exception:
                                    pass
                                i = j
                                break
                i += 1
                
    print("\n=== Parsed and Scaled Coordinates (Normalized -> Pixel) ===")
    for el in raw_markers:
        if "bbox" in el and len(el["bbox"]) == 4:
            xmin, ymin, xmax, ymax = el["bbox"]
            
            px_xmin = int(xmin * width / 1000.0)
            px_ymin = int(ymin * height / 1000.0)
            px_xmax = int(xmax * width / 1000.0)
            px_ymax = int(ymax * height / 1000.0)
            
            x_center = (px_xmin + px_xmax) // 2
            y_center = (px_ymin + px_ymax) // 2
            
            # Absolute click coord (assuming Chrome region starts at left=1927, top=0)
            abs_x = 1927 + x_center
            abs_y = 0 + y_center
            
            print(f"ID: {el.get('id')}, Text: {el.get('text')}")
            print(f"  VLM BBox (0-1000): {el['bbox']}")
            print(f"  Pixel Coords (relative to window): [{px_xmin}, {px_ymin}, {px_xmax}, {px_ymax}]")
            print(f"  Center (relative): ({x_center}, {y_center})")
            print(f"  Absolute Click Coords: ({abs_x}, {abs_y})")

if __name__ == "__main__":
    debug_coords()
