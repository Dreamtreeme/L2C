import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mss
import mss.tools
import pygetwindow as gw
from PIL import Image

from agent.utils.logger import logger
from shared.config import SCREENSHOT_DIR


class PerceptionEngine:
    """
    모니터 화면을 인식하고 분석하는 Perception 엔진입니다.
    mss를 이용한 고속 화면 캡처 및 (추후) OmniParser 연동을 담당합니다.
    """

    def __init__(self):
        self.screenshot_dir = SCREENSHOT_DIR
        self.sct = mss.mss()
        logger.info("PerceptionEngine initialized", screenshot_dir=str(self.screenshot_dir))

    def _get_browser_region(self) -> Optional[Dict[str, int]]:
        """
        열려있는 창 중에서 브라우저(Chrome, Edge, Whale)를 찾아 해당 영역을 반환합니다.
        """
        keywords = ["Chrome", "Edge", "Whale", "크롬", "엣지", "웨일"]
        
        for win in gw.getAllWindows():
            # 최소화되어 있거나 숨겨진 창은 제외
            if not win.visible or win.isMinimized:
                continue
                
            if any(k in win.title for k in keywords):
                # 브라우저를 맨 앞으로 가져오기 (포커스 활성화)
                try:
                    win.activate()
                except Exception as e:
                    logger.debug("Failed to activate window (bring to front)", error=str(e))
                    
                # Windows 10/11의 DWM(Desktop Window Manager)은 
                # 창 주변의 투명한 그림자 영역(약 8px)까지 창 크기로 인식합니다.
                # 배경이 찍히는 것을 막기 위해 이 보이지 않는 테두리를 잘라냅니다.
                border = 8
                
                # 최대화 상태일 때 상단 여백도 조정 필요 (-8로 넘어오는 경우가 많음)
                top = win.top + border if win.isMaximized else win.top
                
                return {
                    "top": top,
                    "left": win.left + border,
                    "width": win.width - (border * 2),
                    "height": win.height - border - (border if win.isMaximized else 0)
                }
        return None

    def capture_screen(self, filename: Optional[str] = None) -> Path:
        """
        브라우저 창 영역(없으면 주 모니터 전체)을 캡처하여 지정된 디렉토리에 저장합니다.
        
        Args:
            filename: 저장할 파일명. 입력하지 않으면 타임스탬프 기반 자동 생성.
            
        Returns:
            저장된 스크린샷 이미지의 절대 경로 (Path 객체)
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_{timestamp}.png"
            
        output_path = self.screenshot_dir / filename
        
        # 1. 브라우저 창 영역 찾기
        region = self._get_browser_region()
        
        try:
            if region:
                # 브라우저만 캡처
                sct_img = self.sct.grab(region)
                logger.debug("Captured browser window only", region=region)
            else:
                # 브라우저를 못 찾으면 모니터 1번 (주 모니터) 전체 캡처
                monitor = self.sct.monitors[1]
                sct_img = self.sct.grab(monitor)
                logger.debug("Browser not found, captured full monitor", monitor=monitor)
                
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(output_path))
            
            logger.info(
                "Screen captured successfully", 
                width=sct_img.width, 
                height=sct_img.height, 
                output_path=str(output_path)
            )
            return output_path
        except Exception as e:
            logger.exception("Failed to capture screen", error=str(e))
            raise

    def analyze_ui(self, image_path: Path) -> Dict[str, Any]:
        """
        캡처된 이미지를 로컬 Ollama (Qwen2.5-VL)에 전송하여 UI 요소를 파싱합니다.
        
        Args:
            image_path: 파싱할 이미지 파일의 경로
            
        Returns:
            UI 마커의 ID, 텍스트, 바운딩 박스(bbox) 목록을 담은 딕셔너리
        """
        import base64
        import json
        import requests
        
        if not image_path.exists():
            logger.error("Image file not found for UI analysis", image_path=str(image_path))
            raise FileNotFoundError(f"Image not found: {image_path}")

        # ----------------------------------------------------
        # 1. Gemini 3.5 Flash를 이용한 고정밀 분석 시도 (권장)
        # ----------------------------------------------------
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                logger.info("Analyzing UI elements via Gemini 3.5 Flash (High Accuracy)")
                with Image.open(image_path) as img:
                    width, height = img.size
                    
                    # Gemini 업로드 속도 및 비용 최적화를 위한 최대 1024px 리사이징
                    max_dim = 1024
                    if width > max_dim or height > max_dim:
                        ratio = max_dim / max(width, height)
                        new_w = int(width * ratio)
                        new_h = int(height * ratio)
                        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        
                        from io import BytesIO
                        buffered = BytesIO()
                        resized_img.save(buffered, format="PNG")
                        img_bytes = buffered.getvalue()
                    else:
                        with open(image_path, "rb") as f:
                            img_bytes = f.read()

                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
                prompt = """
Analyze this UI screenshot of a Korean website. Extract at most 25 most important clickable elements (buttons, inputs, tabs, search/login buttons, job listing cards, search results).
You MUST return a single JSON object with the key "elements".

Each object in the "elements" array must have:
- "id": integer (start from 0)
- "text": visible text or description of the icon/button (e.g. "검색창", "돋보기", "채용공고 타이틀")
- "bbox": [xmin, ymin, xmax, ymax] representing the bounding box, normalized to 0-1000 scale. Note the order: xmin, ymin, xmax, ymax (0-1000).

Example output format:
{
  "elements": [
    {"id": 0, "text": "검색창", "bbox": [350, 50, 450, 100]}
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
                
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                resp_json = response.json()
                text = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                data = json.loads(text)
                raw_markers = data.get("elements", [])
                
                markers = []
                for m in raw_markers:
                    if isinstance(m, dict) and "bbox" in m and len(m["bbox"]) == 4:
                        xmin, ymin, xmax, ymax = m["bbox"]
                        
                        px_xmin = int(xmin * width / 1000.0)
                        px_ymin = int(ymin * height / 1000.0)
                        px_xmax = int(xmax * width / 1000.0)
                        px_ymax = int(ymax * height / 1000.0)
                        
                        # 화면 영역 아웃클리핑 차단
                        px_xmin = max(0, min(width - 1, px_xmin))
                        px_ymin = max(0, min(height - 1, px_ymin))
                        px_xmax = max(0, min(width - 1, px_xmax))
                        px_ymax = max(0, min(height - 1, px_ymax))
                        
                        markers.append({
                            "id": m.get("id", len(markers)),
                            "text": m.get("text", "Unknown"),
                            "bbox": [px_xmin, px_ymin, px_xmax, px_ymax]
                        })
                        
                logger.info("Gemini UI analysis completed successfully", markers_count=len(markers))
                return {
                    "markers": markers,
                    "original_image": str(image_path)
                }
            except Exception as gemini_err:
                logger.warning("Gemini UI analysis failed, falling back to local Ollama", error=str(gemini_err))

        # ----------------------------------------------------
        # 2. 로컬 Ollama (Qwen2.5-VL)를 이용한 Fallback 분석
        # ----------------------------------------------------
        logger.info("Analyzing UI elements via local Ollama (Fallback)", image_path=str(image_path))
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # VRAM 부족 방지를 위한 1024px 리사이징
                max_dim = 1024
                if width > max_dim or height > max_dim:
                    ratio = max_dim / max(width, height)
                    new_w = int(width * ratio)
                    new_h = int(height * ratio)
                    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    from io import BytesIO
                    buffered = BytesIO()
                    resized_img.save(buffered, format="PNG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                else:
                    with open(image_path, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode('utf-8')
                
            prompt = """
Analyze this UI screenshot of a Korean website. Extract at most 25 most important clickable elements (buttons, inputs, tabs, search/login buttons, job listing cards, search results).
당신은 반드시 다음 형식의 JSON 객체 형태로 응답을 작성해야 합니다. 마크다운 코드블록(```json ... ```) 안에 담아서 출력해 주세요.

{
  "elements": [
    {"id": 0, "text": "검색창", "bbox": [xmin, ymin, xmax, ymax]}
  ]
}
Note: bbox should be normalized to 0-1000 scale. Note the order: xmin, ymin, xmax, ymax (0-1000).
"""
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
            logger.info(f"Calling Ollama with model: {model_name}")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "images": [base64_image],
                    "stream": False,
                    # "format": "json" <- 무한 개행 루프를 유발하므로 제거
                    "options": {
                        "num_ctx": 4096,
                        "num_predict": 1024,
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
            
            raw_markers = []
            parse_target = result_text if result_text else thinking_text
            
            if parse_target:
                try:
                    clean_target = parse_target
                    if "```json" in clean_target:
                        clean_target = clean_target.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_target:
                        clean_target = clean_target.split("```")[1].split("```")[0].strip()
                    
                    parsed = json.loads(clean_target)
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
            
            markers = []
            if isinstance(raw_markers, list):
                for m in raw_markers:
                    if isinstance(m, dict) and "bbox" in m and len(m["bbox"]) == 4:
                        xmin, ymin, xmax, ymax = m["bbox"]
                        
                        px_xmin = int(xmin * width / 1000.0)
                        px_ymin = int(ymin * height / 1000.0)
                        px_xmax = int(xmax * width / 1000.0)
                        px_ymax = int(ymax * height / 1000.0)
                        
                        px_xmin = max(0, min(width - 1, px_xmin))
                        px_ymin = max(0, min(height - 1, px_ymin))
                        px_xmax = max(0, min(width - 1, px_xmax))
                        px_ymax = max(0, min(height - 1, px_ymax))
                        
                        markers.append({
                            "id": m.get("id", len(markers)),
                            "text": m.get("text", "Unknown"),
                            "bbox": [px_xmin, px_ymin, px_xmax, px_ymax]
                        })
                    
            logger.info("Ollama UI analysis completed", markers_count=len(markers))
            return {
                "markers": markers,
                "original_image": str(image_path)
            }
            
        except Exception as e:
            logger.error("Failed to analyze UI with Ollama", error=str(e))
            return {"markers": [], "original_image": str(image_path)}
