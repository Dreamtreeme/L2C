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


from agent.tools.som_engine import SomEngine


class PerceptionEngine:
    """
    모니터 화면을 인식하고 분석하는 Perception 엔진입니다.
    mss를 이용한 고속 화면 캡처 및 OmniParser 연동을 담당합니다.
    """

    def __init__(self):
        self.screenshot_dir = SCREENSHOT_DIR
        self.sct = mss.mss()
        self.som_engine = SomEngine()
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.last_region = None
        logger.info("PerceptionEngine initialized with SomEngine", screenshot_dir=str(self.screenshot_dir))

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
            filename = f"screen_{timestamp}.jpg"
            
        output_path = self.screenshot_dir / filename
        
        # 1. 브라우저 창 영역 찾기
        region = self._get_browser_region()
        
        try:
            if region:
                # 브라우저만 캡처
                sct_img = self.sct.grab(region)
                self.scale_x = sct_img.width / region["width"]
                self.scale_y = sct_img.height / region["height"]
                self.last_region = region
                logger.debug("Captured browser window only", region=region, scale_x=self.scale_x, scale_y=self.scale_y)
            else:
                # 브라우저를 못 찾으면 모니터 1번 (주 모니터) 전체 캡처
                monitor = self.sct.monitors[1]
                sct_img = self.sct.grab(monitor)
                self.scale_x = 1.0
                self.scale_y = 1.0
                self.last_region = {
                    "top": monitor["top"],
                    "left": monitor["left"],
                    "width": monitor["width"],
                    "height": monitor["height"]
                }
                logger.debug("Browser not found, captured full monitor", monitor=monitor)
                
            # Convert to PIL Image and save as compressed JPEG
            from PIL import Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            if output_path.suffix.lower() in (".jpg", ".jpeg"):
                img.save(str(output_path), "JPEG", quality=80)
            else:
                img.save(str(output_path))
            
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
        Set-of-Marks (SoM) 기반의 UI 분석 엔진입니다.
        로컬 YOLOv8 및 PaddleOCR로 마킹 이미지를 생성한 뒤 VLM(Gemini/Ollama)을 호출하여
        각 마커 ID의 서비스 상 용도를 캡셔닝하고 물리 좌표와 맵핑하여 반환합니다.
        
        Args:
            image_path: 원본 스크린샷 이미지 경로
            
        Returns:
            UI 마커의 ID, 텍스트, 바운딩 박스(bbox) 목록을 담은 딕셔너리
        """
        import base64
        import json
        import requests
        
        if not image_path.exists():
            logger.error("Image file not found for UI analysis", image_path=str(image_path))
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 1. 로컬 SoM 엔진 실행 (마킹 이미지 합성 및 좌표 추출)
        try:
            marked_filename = f"marked_{image_path.name}"
            marked_path, marker_coords, marker_bboxes, final_elements = self.som_engine.process_image(
                image_path, 
                output_filename=marked_filename
            )
        except Exception as som_err:
            logger.error("Local SoM processing failed", error=str(som_err))
            return {"markers": [], "original_image": str(image_path)}

        skip_vlm_caption = os.getenv("SKIP_VLM_CAPTION", "true").lower() == "true"
        elements = []

        if skip_vlm_caption:
            logger.info("Bypassing VLM captioning node as SKIP_VLM_CAPTION is set to true.")
        else:
            # 2. 마킹된 이미지 로드 및 리사이징 (JPEG 압축 및 VLM 최적화)
            try:
                with Image.open(marked_path) as img:
                    width, height = img.size
                    max_dim = 1024
                    
                    # 필요 시 리사이징 진행
                    if width > max_dim or height > max_dim:
                        ratio = max_dim / max(width, height)
                        new_w = int(width * ratio)
                        new_h = int(height * ratio)
                        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    else:
                        resized_img = img.copy()
                        
                    if resized_img.mode != "RGB":
                        resized_img = resized_img.convert("RGB")
                        
                    from io import BytesIO
                    buffered = BytesIO()
                    resized_img.save(buffered, format="JPEG", quality=80)
                    img_bytes = buffered.getvalue()
                    
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
            except Exception as img_err:
                logger.error("Failed to load and resize marked image", error=str(img_err))
                return {"markers": [], "original_image": str(image_path)}

            # 3. VLM 프롬프트 작성 (ID 매핑 요청 - 토큰 길이 및 속도 최적화 버전)
            prompt = """
Analyze this UI screenshot of a Korean website, which has numbered markers on it (like [0], [1], [2], ...).
Describe ONLY the most important clickable/interactable elements (e.g. GNB menu items, major buttons, input fields, search results, tabs).

Optimization rules:
1. Focus ONLY on interactive/clickable elements. Ignore background static texts, tiny decorations, or unidentifiable symbols.
2. Keep descriptions extremely short and concise (e.g., 2-4 words maximum, like "검색창", "구글 로그인", "데이터 분석가 채용").
3. Limit the response to at most 35-40 of the most significant elements to keep it compact.

You MUST return a single JSON object with the key "elements".
Each object in the "elements" array must have:
- "id": integer corresponding to the marker number in the image
- "text": short description of the element (e.g. "구글 로그인")

Example output format:
{
  "elements": [
    {"id": 0, "text": "검색창"},
    {"id": 1, "text": "회원가입"}
  ]
}
"""

            api_key = os.getenv("GEMINI_API_KEY")

            # 4. Gemini 3.5 Flash 호출 시도
            if api_key:
                try:
                    logger.info("Captioning UI elements via Gemini 3.5 Flash SoM...")
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {
                                    "inlineData": {
                                        "mimeType": "image/jpeg",
                                        "data": base64_image
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
                    elements = data.get("elements", [])
                    logger.info("Gemini SoM captioning completed successfully", elements_count=len(elements))
                except Exception as gemini_err:
                    logger.warning("Gemini SoM captioning failed, falling back to local Ollama", error=str(gemini_err))

            # 5. 로컬 Ollama (Qwen2.5-VL) Fallback 호출 시도
            if not elements:
                logger.info("Captioning UI elements via local Ollama SoM (Fallback)...")
                try:
                    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "images": [base64_image],
                            "stream": False,
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
                    parse_target = result_text if result_text else thinking_text
                    
                    if parse_target:
                        clean_target = parse_target
                        if "```json" in clean_target:
                            clean_target = clean_target.split("```json")[1].split("```")[0].strip()
                        elif "```" in clean_target:
                            clean_target = clean_target.split("```")[1].split("```")[0].strip()
                        
                        try:
                            parsed = json.loads(clean_target)
                            if isinstance(parsed, dict) and "elements" in parsed:
                                elements = parsed["elements"]
                            elif isinstance(parsed, list):
                                elements = parsed
                        except Exception:
                            pass
                    logger.info("Ollama SoM captioning completed", elements_count=len(elements))
                except Exception as ollama_err:
                    logger.error("Failed to caption UI elements via Ollama", error=str(ollama_err))

        # 6. 매핑 정보 병합 및 안전한 Fallback 매핑 (VLM 누락 마커 처리)
        id_to_text = {}
        if skip_vlm_caption:
            for marker_id, elem in enumerate(final_elements):
                local_text = elem.get("text", "")
                elem_type = elem.get("type", "element")
                if elem_type == "text" and local_text:
                    id_to_text[marker_id] = local_text
                else:
                    id_to_text[marker_id] = f"상호작용 가능한 요소 ({elem_type})"
        else:
            if elements:
                for elem in elements:
                    if isinstance(elem, dict) and "id" in elem:
                        try:
                            id_to_text[int(elem["id"])] = elem.get("text", "상호작용 가능한 요소")
                        except ValueError:
                            continue

        markers = []
        for marker_id, bbox in marker_bboxes.items():
            text = id_to_text.get(marker_id, "상호작용 가능한 요소 (미식별)")
            markers.append({
                "id": marker_id,
                "text": text,
                "bbox": bbox
            })

        logger.info("UI analysis pipeline complete", final_markers_count=len(markers))
        return {
            "markers": markers,
            "original_image": str(image_path),
            "marked_image": str(marked_path)
        }
