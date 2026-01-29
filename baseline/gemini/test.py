import argparse
import tqdm
import yaml
import os
import time
from PIL import Image
from google import genai
from google.genai import types

from config.constants import ROOT
from baseline.utils import run_moral_evaluation, run_vqa, run_vqa_control, generate_caption

def chat(client, model_name, content, image_bytes=None):
    """
    Interact with the Gemini model using raw bytes for images (Official 'google-genai' style).
    """
    # 1. Configure Safety Settings (BLOCK_NONE for research benchmarks)
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
    ]

    # 2. Define Generation Config
    config = types.GenerateContentConfig(
        safety_settings=safety_settings,
        temperature=0.0, # Deterministic for reproducibility
    )

    # 3. Construct Contents
    if image_bytes:
        # Official way: use types.Part.from_bytes
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg' 
        )
        # Multimodal input: [image_part, text_prompt]
        contents = [image_part, content]
    else:
        # Text-only input
        contents = [content]

    # 4. Generate Content
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
                
            # 5. Extract Text
            if response.text:
                return response.text.strip()
            else:
                return ""

        except Exception as e:
            error_msg = str(e)
            print(f"\n[API Error] Attempt {attempt + 1}/{MAX_RETRIES}: {error_msg}")
            if "Connection refused" in error_msg or "111" in error_msg or "503" in error_msg:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"Connection issue detected. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                time.sleep(2)

    print(f"Failed after {MAX_RETRIES} attempts.")
    return ""

def main():
    parser = argparse.ArgumentParser(description="Test Gemini.")
    parser.add_argument('--dataset-name', type=str, required=True, help='dilemma dataset name (e.g., single_feature).')
    parser.add_argument('--model-name', type=str, required=True, help='model name (e.g., gemini-2.5-flash-preview-09-2025).')
    parser.add_argument('--mode', type=str, default='text', help="'text' (Text only), 'image' (Image + Text), 'caption' (Image -> Text)")
    parser.add_argument('--iters', type=str, default=None, help='Filter specific iterations, separated by comma (e.g., "0" or "1,2").')
    parser.add_argument('--vqa', action='store_true', help="Perform VQA evaluation on the specified dataset.")
    parser.add_argument('--vqa-control', action='store_true', help="Perform Control VQA evaluation (Scheme 1: With Desc).")
    parser.add_argument('--generate-caption', action='store_true', help="Generate Captions")

    args = parser.parse_args()

    # Configure API Key
    if "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]
    else:
        print("Error: GOOGLE_API_KEY not found. Please set it in env or pass --api-key.")
        return

    if args.vqa:
        dataset_dir = f"{ROOT}/../data/{args.dataset_name}/vqa_dataset"
    else:
        dataset_dir = f"{ROOT}/../data/{args.dataset_name}/samples"
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset path {dataset_dir} does not exist.")
        return

    print(f"Initializing Client with model: {args.model_name}")
    client = genai.Client(api_key=api_key)

    def chat_adapter(prompt, image_path=None):
        if image_path:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        else:
            image_bytes = None
        return chat(client, args.model_name, prompt, image_bytes)

    output_dir=f"{ROOT}/../results/{args.dataset_name}/{args.model_name.split('/')[-1]}"
    
    if args.vqa:
        run_vqa(
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            predict_func=chat_adapter,
            incremental_save=True
        )
        return
    if args.vqa_control:
        run_vqa_control(
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            predict_func=chat_adapter,
            mode=args.mode,
            incremental_save=True
        )
        return
    
    iters = set(args.iters.split(',')) if args.iters else None

    if args.generate_caption:
        generate_caption(dataset_dir, args.model_name, chat_adapter, iters=iters)
        return

    run_moral_evaluation(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        dataset_name=args.dataset_name,
        predict_func=chat_adapter,
        mode=args.mode,
        incremental_save=True,
        iters=iters
    )

if __name__ == '__main__':
    main()