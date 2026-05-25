import json
import base64
import requests
import time
from pathlib import Path
from PIL import Image

def run_limit_simulation():
    image_path = r"C:\Users\psg\Desktop\L2C\data\screenshots\screen_20260520_073042.png"
    
    img = Image.open(image_path)
    width, height = img.size
    print(f"Original Resolution: {width}x{height}")
    
    # 1. Resize to 1024px (Restored high-accuracy baseline)
    max_dim = 1024
    ratio = max_dim / max(width, height)
    new_w = int(width * ratio)
    new_h = int(height * ratio)
    print(f"Resizing to {new_w}x{new_h} for high-detail local test...")
    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    import io
    buffered = io.BytesIO()
    resized_img.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 2. Prepare Prompt (Natural language prompt asking for JSON inside code blocks, no format='json')
    prompt = """
Analyze this UI screenshot of a Korean website. Extract at most 8 most important clickable elements (buttons, inputs, tabs, search/login buttons).
당신은 반드시 다음 형식의 JSON 객체 형태로 응답을 작성해야 합니다. 마크다운 코드블록(```json ... ```) 안에 담아서 출력해 주세요.

{
  "elements": [
    {"id": 0, "text": "검색창", "bbox": [xmin, ymin, xmax, ymax]}
  ]
}
Note: bbox should be normalized to 0-1000 scale.
"""
    
    print("\n[Simulation] Running Qwen2.5-VL 7B with 768px image and NO format='json' constraint...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5vl:7b",
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                # "format": "json" <- Removed!
                "options": {
                    "num_ctx": 2048,      # Reduced to save VRAM and prevent CPU offloading
                    "num_predict": 1024,
                    "temperature": 0.1
                }
            },
            timeout=120
        )
        response.raise_for_status()
        elapsed = time.time() - start_time
        
        resp_json = response.json()
        result_text = resp_json.get("response", "").strip()
        
        print(f"\nCompleted in {elapsed:.2f} seconds.")
        print("\n=== Raw Output ===")
        print(result_text)
        print("==================")
        
        # Let's see if we can parse it using stack parser
        raw_markers = []
        i = 0
        while i < len(result_text):
            if result_text[i] == '{':
                stack = 0
                for j in range(i, len(result_text)):
                    if result_text[j] == '{':
                        stack += 1
                    elif result_text[j] == '}':
                        stack -= 1
                        if stack == 0:
                            candidate = result_text[i:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    if "bbox" in obj:
                                        raw_markers.append(obj)
                                    elif "elements" in obj:
                                        raw_markers.extend(obj["elements"])
                            except Exception:
                                pass
                            i = j
                            break
            i += 1
            
        print("\n=== Parsed Coordinates (Normalized -> Original Pixel) ===")
        for el in raw_markers:
            if "bbox" in el and len(el["bbox"]) == 4:
                xmin, ymin, xmax, ymax = el["bbox"]
                px_xmin = int(xmin * width / 1000.0)
                px_ymin = int(ymin * height / 1000.0)
                px_xmax = int(xmax * width / 1000.0)
                px_ymax = int(ymax * height / 1000.0)
                
                x_center = (px_xmin + px_xmax) // 2
                y_center = (px_ymin + px_ymax) // 2
                
                print(f"ID: {el.get('id')}, Text: {el.get('text')}")
                print(f"  Normalized BBox: {el['bbox']}")
                print(f"  Original Pixel Coords: [{px_xmin}, {px_ymin}, {px_xmax}, {px_ymax}]")
                print(f"  Center: ({x_center}, {y_center}) -> Absolute: ({1927 + x_center}, {y_center})")
                
    except Exception as e:
        print("Simulation failed:", e)

if __name__ == "__main__":
    run_limit_simulation()
