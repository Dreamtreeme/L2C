import os
import json
import base64
import dotenv
from pathlib import Path
from PIL import Image

# Load environment
dotenv.load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

def test_gemini_perception():
    image_path = r"C:\Users\psg\Desktop\L2C\data\screenshots\screen_20260520_073042.png"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found!")
        return
        
    print("Using GEMINI_API_KEY:", api_key[:10] + "...")
    
    # 1. Load image and convert to base64
    img = Image.open(image_path)
    width, height = img.size
    print(f"Original Resolution: {width}x{height}")
    
    # Resize to max 1024 to make API call fast but keep details sharp
    max_dim = 1024
    if width > max_dim or height > max_dim:
        ratio = max_dim / max(width, height)
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        print(f"Resizing to {new_w}x{new_h} for Gemini...")
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    import io
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    # 2. Call Gemini 3.5 Flash
    # We will use raw HTTP requests to avoid library version mismatches.
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
    
    prompt = """
Analyze this UI screenshot of a Korean website. Extract at most 8 most important clickable elements (buttons, inputs, tabs, search/login buttons).
You MUST return a single JSON object with the key "elements".

Each object in the "elements" array must have:
- "id": integer (start from 0)
- "text": visible text or description of the icon (e.g. "검색창", "돋보기", "탐색")
- "bbox": [ymin, xmin, ymax, xmax] representing the bounding box, normalized to 0-1000 scale. Note the order: ymin, xmin, ymax, xmax (0-1000).

Example output format:
{
  "elements": [
    {"id": 0, "text": "로그인/회원가입", "bbox": [50, 850, 80, 950]}
  ]
}
"""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(img_bytes).decode("utf-8")
                    }
                }
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }
    
    print("Calling Gemini API...")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    resp_json = response.json()
    
    print("=== Gemini API Response ===")
    text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
    print(text)
    print("===========================")
    
    # Let's map it back to original resolution and print
    data = json.loads(text)
    elements = data.get("elements", [])
    
    print("\n=== Mapped Coordinates to Original Image ===")
    for el in elements:
        ymin, xmin, ymax, xmax = el["bbox"] # Note: Gemini standard is usually [ymin, xmin, ymax, xmax] or whatever the prompt says.
        # Let's map normalized coordinates:
        px_xmin = int(xmin * width / 1000.0)
        px_ymin = int(ymin * height / 1000.0)
        px_xmax = int(xmax * width / 1000.0)
        px_ymax = int(ymax * height / 1000.0)
        
        x_center = (px_xmin + px_xmax) // 2
        y_center = (px_ymin + px_ymax) // 2
        
        abs_x = 1927 + x_center
        abs_y = 0 + y_center
        
        print(f"ID: {el['id']}, Text: {el['text']}")
        print(f"  BBox: {el['bbox']}")
        print(f"  Pixel Coords (relative): [{px_xmin}, {px_ymin}, {px_xmax}, {px_ymax}]")
        print(f"  Center: ({x_center}, {y_center})")
        print(f"  Absolute Click Coords: ({abs_x}, {abs_y})")

if __name__ == "__main__":
    test_gemini_perception()
