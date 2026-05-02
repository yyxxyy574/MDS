from itertools import product
import os
import random
import yaml
import argparse
import tqdm

from config.constants import CHARACTER, DILEMMA, SCENE, ROOT
from data.utils import sample_same_quantities, sample_positions, place_characters
from env.gui.canvas import Canvas

QUANTITY_LEVEL = [(1, 1), (1, 2), (1, 5), (1, 10)]
QUANTITY_LEVEL_INVERSE = [(2, 1), (5, 1), (10, 1)]
MAX_QUANTITY = {
    'trolley': 10,
    'footbridge': 10,
    'vaccine_policy': 90,
    'environmental_policy': 90, 
    'lifeboat': 10, 
    'crying_baby': 10, 
    'shark_attack': 10, 
    'transplant': 5, 
    'terrorist': 10,
}

tileset_character_path = f"{ROOT}/../draw_map/tilesets/character.png"
tileset_animal_path = f"{ROOT}/../draw_map/tilesets/animal.png"
tileset_arrow_path = f"{ROOT}/../draw_map/tilesets/arrow.png"
tileset_emoji_path = f"{ROOT}/../draw_map/tilesets/emoji.png"
tileset_emoji_box_path = f"{ROOT}/../draw_map/tilesets/emoji_box.png"

def process_instance(dimension, dilemma, dilemma_instance, seed, iter=0, inverse=False):
    if dilemma not in MAX_QUANTITY:
        print(f"Not for {dilemma}, skip.")
        return

    random.seed(seed + iter * 184)

    quantity_levels = QUANTITY_LEVEL_INVERSE if inverse else QUANTITY_LEVEL
    mode_str = "INVERSE" if inverse else "NORMAL"

    print(f"\nProcessing dilemma: {dilemma} (Mode: {mode_str})...")

    if 'scene' not in DILEMMA[dimension][dilemma]:
        print(f"  Skipping '{dilemma}': no scene specified.")
        return
    
    scene = DILEMMA[dimension][dilemma]['scene']

    tmx_files = [f for f in os.listdir(f"draw_map/maps/{scene}") if f.endswith(".tmx")]
    if not tmx_files:
        print(f"  Error: TMX file not found for scene '{scene}'. Skipping.")
        return

    # Identify the two groups of characters to apply the ratio to (excluding agent)
    target_characters = [
        char for char in DILEMMA[dimension][dilemma][dilemma_instance]['character']
        if char != 'agent' and 'is_related' not in DILEMMA[dimension][dilemma][dilemma_instance]['character'][char]
    ]
    
    # Handle cases where we might catch extra utility characters, usually strictly 2 groups for these dilemmas
    if len(target_characters) < 2:
        print(f"  Warning: Less than 2 target characters found for dilemma '{dilemma}'. Found: {target_characters}. Skipping.")
        return

    # Identify the two groups of characters to apply the ratio to (excluding agent)
    # These are the characters whose quantities we want to manipulate
    target_characters = [
        char for char in DILEMMA[dimension][dilemma][dilemma_instance]['character']
        if char != 'agent' and 'is_related' not in DILEMMA[dimension][dilemma][dilemma_instance]['character'][char]
    ]

    if len(target_characters) < 2:
        print(f"  Warning: Less than 2 target characters found for dilemma '{dilemma}'. Found: {target_characters}. Skipping.")
        return
    
    for (r1, r2) in quantity_levels:

        save_dir = f"{ROOT}/../data/quantity/samples/{dimension}/{dilemma}/{dilemma_instance}/{r1}vs{r2}"
        save_dir_no_desc = f"{ROOT}/../data/quantity/samples_no_desc/{dimension}/{dilemma}/{dilemma_instance}/{r1}vs{r2}"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_no_desc, exist_ok=True)
        
        valid_combinations = []
        multiplier = 1
        limit = MAX_QUANTITY.get(dilemma, 10)

        while True:
            # Conceptual total quantities (e.g., lives at stake)
            total_q1 = r1 * multiplier
            total_q2 = r2 * multiplier
            
            # Check maximum constraint (on Total Quantity)
            if max(total_q1, total_q2) > limit:
                break
            
            if dilemma in {'trolley', 'footbridge', 'vaccine_policy','environmental_policy'}:
                if 'same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][target_characters[0]]:
                    valid_combinations.append({target_characters[0]: total_q2 - 1, target_characters[1]: total_q1})
                else:
                    valid_combinations.append({target_characters[0]: total_q2, target_characters[1]: total_q1})
            else:
                if 'quantity' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][target_characters[0]] and total_q1 not in DILEMMA[dimension][dilemma][dilemma_instance]['character'][target_characters[0]]['quantity']:
                    break
                if 'same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][target_characters[1]]:
                    valid_combinations.append({target_characters[0]: total_q1, target_characters[1]: total_q2 - 1})
                else:
                    valid_combinations.append({target_characters[0]: total_q1, target_characters[1]: total_q2})
            
            multiplier += 1
        
        if not valid_combinations:
            print(f"  No valid quantity combinations for ratio {r1}:{r2} in dilemma '{dilemma}'. Skipping.")
            continue

        k = 0
        while k < 5:
            if k == 4:
                combination = valid_combinations[0]
            else:
                combination = random.choice(valid_combinations)

            features = {}
            color = random.choice(CHARACTER['color'])
            for char in target_characters:
                features[char] = {'color': color}
            features['agent'] = {'color': color}

            max_retries = 50
            success = False
            
            for attempt in range(max_retries):
                tmx_file = random.choice(tmx_files)
                tmx_path = f"{ROOT}/../draw_map/maps/{scene}/{tmx_file}"
                with open(tmx_path.rpartition(".")[0] + ".yaml", "r", encoding="utf-8") as f:
                    scene_config = yaml.safe_load(f)
                for zone in scene_config:
                    if 'pos' in SCENE[scene][zone]:
                        SCENE[scene][zone]['pos'] = scene_config[zone]
                    elif 'scope' in SCENE[scene][zone]:
                        SCENE[scene][zone]['scope'] = scene_config[zone]
                canvas = Canvas(tileset_character_path, tileset_animal_path, tileset_arrow_path, tileset_emoji_path, tileset_emoji_box_path, tmx_path=tmx_path)
                canvas.reset()

                character_description = {}
                character_description['tmx_path'] = tmx_path

                character_quantities = {}
                sample_same_quantities(scene, dimension, dilemma, dilemma_instance, character_description, character_quantities)
                for char in combination:
                    character_quantities[char] = combination[char]
                    character_description[f'{char}_quantity'] = combination[char]

                characters_by_place = {}
                fail = sample_positions(scene, dimension, dilemma, dilemma_instance, character_description, character_quantities, characters_by_place)
                if not fail:
                    place_characters(scene, dimension, dilemma, dilemma_instance, character_description, characters_by_place, [], features, canvas)
                    description = canvas.load_description(DILEMMA[dimension][dilemma][dilemma_instance]['description'], character_description, rewrite=True)
                    canvas.render(f"{save_dir_no_desc}/{iter}.jpg", load_description=False)
                    canvas.render(f"{save_dir}/{iter}.jpg", load_description=True)
                    character_description['description'] = description
                    with open(f"{save_dir}/{iter}.yaml", 'w', encoding='utf-8') as f:
                        yaml.dump(character_description, f, allow_unicode=True)
                    print(f"  Generated image: {save_dir}/{iter}.jpg")
                    success = True
        
                if success:
                    break
            
            if success:
                break
            k += 1
        
        if not success:
            print(f"  Failed to generate valid map for ratio {r1}:{r2} in dilemma '{dilemma}' after {max_retries} attempts.")

def main():
    parser = argparse.ArgumentParser(description="Generate dilemma image data for a specific dilemma instance.")
    parser.add_argument('--dimension', type=str, required=True, help='The dilemma dimension (e.g., care-care).')
    parser.add_argument('--dilemma', type=str, required=True, help='The specific dilemma name (e.g., trolley).')
    parser.add_argument('--dilemma-instance', type=str, required=True, help='The specific dilemma instance (e.g., trolley_0_0_0).')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iter', type=int, required=True)
    parser.add_argument('--inverse', action='store_true', help="Generate inverse scenarios (Sacrifice > Saved)")

    args = parser.parse_args()
    
    print(f"--- Starting task for: {args.dimension} / {args.dilemma} / {args.dilemma_instance} / {args.iter} ---")
    process_instance(args.dimension, args.dilemma, args.dilemma_instance, args.seed, args.iter, args.inverse)
    print(f"--- Finished task for: {args.dimension} / {args.dilemma} / {args.dilemma_instance} / {args.iter} ---")

if __name__ == '__main__':
    main()
