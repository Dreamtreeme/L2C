import torch
import time
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image

def run_hf_test():
    image_path = r"C:\Users\psg\Desktop\L2C\data\screenshots\screen_20260520_073042.png"
    if not os.path.exists(image_path):
        print(f"Error: Screenshot not found at {image_path}")
        return

    print("=== Hugging Face Local VLM Test (4-bit + 512px Token Compression) ===")
    
    # 1. Configure 4-bit quantization (Saves VRAM to keep execution entirely on GPU)
    print("[1/4] Configuring bitsandbytes 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 2. Load model and processor
    print("[2/4] Loading model 'Qwen/Qwen2.5-VL-3B-Instruct' in 4-bit (SDPA)...")
    start_load = time.time()
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"  # Built-in PyTorch Scaled Dot Product Attention
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    load_time = time.time() - start_load
    print(f"Model loaded successfully in {load_time:.2f} seconds.")

    # 3. Prepare Prompt and Input
    print("[3/4] Preparing input image and prompt template...")
    prompt = """
Analyze this UI screenshot of a Korean website. Extract at most 8 most important clickable elements (buttons, inputs, tabs, search/login buttons).
For each element, detect:
1. "id": incremental index from 0
2. "text": a short label of the element
3. "bbox": bounding box in [xmin, ymin, xmax, ymax] format (normalized 0-1000)

Return your output STRICTLY in the following JSON format within a single ```json code block. Do NOT add any extra text or conversation.
```json
{
  "elements": [
    {"id": 0, "text": "Label 1", "bbox": [xmin, ymin, xmax, ymax]},
    {"id": 1, "text": "Label 2", "bbox": [xmin, ymin, xmax, ymax]}
  ]
}
```
Note: bbox should be normalized to 0-1000 scale.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process vision inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Enforce max_pixels dynamically at processor level (visual token compression)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        min_pixels=256*256,
        max_pixels=768*768  # Sweet spot between detail and speed
    )
    inputs = inputs.to("cuda")

    # 4. Generate Response
    print("[4/4] Running inference (GPU accelerated)...")
    start_infer = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    infer_time = time.time() - start_infer
    print(f"\nCompleted in {infer_time:.2f} seconds.")
    print("\n=== Raw Output ===")
    print(output_text[0])
    print("==================")

if __name__ == "__main__":
    run_hf_test()
