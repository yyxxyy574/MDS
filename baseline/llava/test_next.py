import argparse
import tqdm
import torch
import yaml
import os
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig

from config.constants import ROOT
from baseline.utils import run_moral_evaluation, run_vqa, run_vqa_control, generate_caption

def chat(model, processor, content, image=None):
    if image is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": image,
                    },
                    {"type": "text", "text": content},
                ],
            }
        ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0] if output_text else ""
    
def main():
    parser = argparse.ArgumentParser(description="Test LLaVA.")
    parser.add_argument('--dataset-name', type=str, required=True, help='dilemma dataset name (e.g., single_feature).')
    parser.add_argument('--model-name', type=str, required=True, help='vlm/llm model name (e.g., models/deepseek-vl-7b-chat).')
    parser.add_argument('--mode', type=str, default='text', help="'text' (Text only), 'image' (Image + Text), 'caption' (Image -> Text)")
    parser.add_argument('--quantize', action='store_true', help="whether to quantize the model.")
    parser.add_argument('--iters', type=str, default=None, help='Filter specific iterations, separated by comma (e.g., "0" or "1,2").')
    parser.add_argument('--vqa', action='store_true', help="Perform VQA evaluation on the specified dataset.")
    parser.add_argument('--vqa-control', action='store_true', help="Perform Control VQA evaluation (Scheme 1: With Desc).")
    parser.add_argument('--generate-caption', action='store_true', help="Generate Captions")
    
    args = parser.parse_args()

    if args.vqa:
        dataset_dir = f"{ROOT}/../data/{args.dataset_name}/vqa_dataset"
    else:
        dataset_dir = f"{ROOT}/../data/{args.dataset_name}/samples"
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset path {dataset_dir} does not exist.")
        return

    print(f"Loading model: {args.model_name}")
    
    model_load_kwargs = {
        "device_map": "auto"
    }
    if args.quantize:
        print("Loading model with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["quantization_config"] = quantization_config
        model_load_kwargs["dtype"] = torch.bfloat16
    else:
        print("Loading model in default precision...")
        model_load_kwargs["dtype"] = "auto"

    print(args.model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name, **model_load_kwargs
    )
    processor = LlavaNextProcessor.from_pretrained(args.model_name)

    def chat_adapter(prompt, image_path=None):
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = None
        return chat(model, processor, prompt, image=image)
    
    output_dir=f"{ROOT}/../results/{args.dataset_name}/{args.model_name.split('/')[-1]}"

    if args.vqa:
        run_vqa(
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            predict_func=chat_adapter,
            incremental_save=False
        )
        return
    if args.vqa_control:
        run_vqa_control(
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            predict_func=chat_adapter,
            incremental_save=False
        )
        return

    if args.generate_caption:
        generate_caption(dataset_dir, args.model_name, chat_adapter)
        return
    
    iters = set(args.iters.split(',')) if args.iters else None
    run_moral_evaluation(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        dataset_name=args.dataset_name,
        predict_func=chat_adapter,
        mode=args.mode,
        incremental_save=False,
        iters=iters
    )

if __name__ == '__main__':
    main()
