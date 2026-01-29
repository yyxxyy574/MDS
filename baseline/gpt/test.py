import argparse
import tqdm
import yaml
import os
import time
import requests
import base64

from config.constants import ROOT
from baseline.utils import run_moral_evaluation, run_vqa, run_vqa_control, generate_caption

def chat(api_key, model_name, content, image_bytes=None, max_tokens=128):
    """
    Interact with the OpenAI API (GPT-4o).
    Args:
        max_tokens (int): Maximum number of tokens to generate. 
    """
    # Changed to OpenAI official endpoint
    invoke_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Construct Messages (Multimodal format)
    if image_bytes:
        # Encode bytes to base64 string
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Construct multimodal content list
        message_content = [
            {
                "type": "text",
                "text": content
            },
            {
                "type": "image_url",
                "image_url": {
                    # OpenAI supports "detail": "auto" | "low" | "high". 
                    # Default is auto. Keeping it simple for consistency.
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    else:
        # Text-only mode
        message_content = content

    payload = {
        "model": model_name, # e.g., "gpt-4o-2024-08-06"
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ],
        "temperature": 0.0, # Deterministic sampling
        "top_p": 1.0,
        "max_tokens": max_tokens, # Dynamic max_tokens based on prompt type
        "stream": False
        # Note: OpenAI API does not expose parameters to explicitly disable safety filters 
        # (unlike Google's safety_settings). Safety behaviors are handled server-side.
    }

    # 2. Generate Content with Retry Logic
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
            
            # Handle non-200 HTTP status
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"\n[API Error] Attempt {attempt + 1}/{MAX_RETRIES}: {error_msg}")
                
                # Retry on rate limits (429) or server errors (5xx)
                if response.status_code in [429, 500, 503, 504]:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"Server busy or rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Client errors (400) usually shouldn't be retried blindly (e.g., policy violations)
                    # For GPT-4o, 400 might indicate a safety refusal at the request level
                    time.sleep(2)
                    continue

            # 3. Extract Text
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                # Check for refusal finish_reason if needed, though usually content is empty then
                content_text = response_json["choices"][0]["message"]["content"]
                return content_text.strip() if content_text else ""
            else:
                print(f"Unexpected response format: {response_json}")
                return ""

        except Exception as e:
            error_msg = str(e)
            print(f"\n[Exception] Attempt {attempt + 1}/{MAX_RETRIES}: {error_msg}")
            time.sleep(2)

    print(f"Failed after {MAX_RETRIES} attempts.")
    return ""

def main():
    parser = argparse.ArgumentParser(description="Test OpenAI GPT Models.")
    parser.add_argument('--dataset-name', type=str, required=True, help='dilemma dataset name (e.g., single_feature).')
    parser.add_argument('--model-name', type=str, required=True, help='model name (e.g., gpt-4o-mini-2024-07-18).')
    parser.add_argument('--mode', type=str, default='text', help="'text' (Text only), 'image' (Image + Text), 'caption' (Image -> Text)")
    parser.add_argument('--iters', type=str, default=None, help='Filter specific iterations, separated by comma (e.g., "0" or "1,2").')
    parser.add_argument('--vqa', action='store_true', help="Perform VQA evaluation on the specified dataset.")
    parser.add_argument('--vqa-control', action='store_true', help="Perform Control VQA evaluation (Scheme 1: With Desc).")
    parser.add_argument('--generate-caption', action='store_true', help="Generate Captions")

    args = parser.parse_args()

    # Configure API Key (Changed to OPENAI_API_KEY)
    if "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
    else:
        print("Error: OPENAI_API_KEY not found. Please set it in env.")
        return

    if args.vqa:
        dataset_dir = f"{ROOT}/../data/{args.dataset_name}/vqa_dataset"
    else:
        dataset_dir = f"{ROOT}/../data/{args.dataset_name}/samples"
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset path {dataset_dir} does not exist.")
        return

    print(f"Initializing Test with model: {args.model_name}")

    def chat_adapter(prompt, image_path=None):
        """
        Adapter to handle image reading and token logic before calling the API.
        """
        if image_path:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        else:
            image_bytes = None
        # Ensure max_tokens matches the logic for detailed descriptions vs short answers
        if 'describe' in prompt.lower(): 
            return chat(api_key, args.model_name, prompt, image_bytes, max_tokens=2048)
        else:
            return chat(api_key, args.model_name, prompt, image_bytes)

    # Output directory logic remains the same
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