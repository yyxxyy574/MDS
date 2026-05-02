from itertools import product
import os
import random
import yaml
import argparse
import tqdm

from config.constants import CHARACTER, DILEMMA, SCENE, ROOT
from data.utils import sample_same_quantities, sample_positions, place_characters
from env.gui.canvas import Canvas

tileset_character_path = f"{ROOT}/../draw_map/tilesets/character.png"
tileset_animal_path = f"{ROOT}/../draw_map/tilesets/animal.png"
tileset_arrow_path = f"{ROOT}/../draw_map/tilesets/arrow.png"
tileset_emoji_path = f"{ROOT}/../draw_map/tilesets/emoji.png"
tileset_emoji_box_path = f"{ROOT}/../draw_map/tilesets/emoji_box.png"

def generate_single_feature_samples(dimension, dilemma, dilemma_instance, available_features=['species', 'color', 'gender', 'profession', 'age', 'wealth', 'fitness', 'education']):
    
    available_values = {}
    for character in DILEMMA[dimension][dilemma][dilemma_instance]['character']:
        if 'is_related'in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character] and not DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['is_related']:
                continue
        
        available_values[character] = {}
        for feature in available_features:
            if feature in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character] and 'none' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character][feature]:
                available_values[character][feature] = []
                continue

            if feature in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                available_values[character][feature] = []
                for value in CHARACTER[feature]:
                    if value in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character][feature] or (isinstance(CHARACTER[feature], dict) and set(CHARACTER[feature][value]) & set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character][feature])):
                        available_values[character][feature].append(value)
            else:
                if isinstance(CHARACTER[feature], list):
                    available_values[character][feature] = CHARACTER[feature]
                else:
                    available_values[character][feature] = list(CHARACTER[feature].keys())
    characters = list(available_values.keys())

    feature_samples = {}
    for feature in available_features:
        if feature == 'species':
            if len(characters) == 1:
                continue
            elif len(characters) == 2:
                feature_samples[feature] = {}
                for species_value in available_values[characters[1]]['species']:
                    feature_instance = {}
                    feature_instance['agent'] = {}
                    feature_instance['agent']['species'] = 'human'
                    feature_instance[characters[1]] = {}
                    feature_instance[characters[1]]['species'] = species_value
                    feature_samples[feature][f'human_{species_value}_'] = [feature_instance]
            else:
                 feature_samples[feature] = {}
                 for (species_value_1, species_value_2) in list(product(available_values[characters[1]]['species'], available_values[characters[2]]['species'])):
                    if ('same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][characters[1]] and species_value_1 != 'human') or ('same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][characters[2]] and species_value_2 != 'human'):
                        continue
                    feature_instance = {}
                    feature_instance['agent'] = {}
                    feature_instance['agent']['species'] = 'human'
                    feature_instance[characters[1]] = {}
                    feature_instance[characters[1]]['species'] = species_value_1
                    feature_instance[characters[2]] = {}
                    feature_instance[characters[2]]['species'] = species_value_2
                    feature_samples[feature][f'human_{species_value_1}_{species_value_2}_'] = [feature_instance]
        else:
            if (len(characters) == 2 and len(available_values[characters[1]]['species']) == 1 and available_values[characters[1]]['species'][0] == 'non-human') or (len(characters) == 3 and len(available_values[characters[2]]['species']) == 1 and available_values[characters[2]]['species'][0] == 'non-human'):
                continue
            
            if feature == 'color':
                if len(characters) == 1:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[0]][feature]))
                elif len(characters) == 2:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[1]][feature]))
                else:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[1]][feature], available_values[characters[2]][feature]))
                feature_samples[feature] = {}
                for value in values:
                    value_name = ""
                    feature_instance = {}
                    for i in range(len(characters)):
                        if ('same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][characters[i]] and value[i] != value[0]):
                            feature_instance = {}
                            break
                        feature_instance[characters[i]] = {}
                        feature_instance[characters[i]]['species'] = 'human'
                        feature_instance[characters[i]][feature] = value[i]
                        value_name += f'{value[i]}_'
                    if len(feature_instance) > 0:
                        feature_samples[feature][value_name] = [feature_instance]
            
            elif feature == 'profession':
                if len(characters) == 1:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[0]][feature]))
                    gender_color_values = list(product(list(product(available_values[characters[0]]['color'], available_values[characters[0]]['gender'])), list(product(available_values[characters[0]]['color'], available_values[characters[0]]['gender']))))
                elif len(characters) == 2:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[1]][feature]))
                    gender_color_values = list(product(list(product(available_values[characters[0]]['color'], available_values[characters[0]]['gender'])), list(product(available_values[characters[1]]['color'], available_values[characters[1]]['gender']))))
                else:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[1]][feature], available_values[characters[2]][feature]))
                    gender_color_values = list(product(list(product(available_values[characters[0]]['color'], available_values[characters[0]]['gender'])), list(product(available_values[characters[1]]['color'], available_values[characters[1]]['gender']))))

                feature_samples[feature] = {}
                for value in values:
                    for gender_color_value in gender_color_values:
                        value_name = ""
                        feature_instance = {}
                        for i in range(len(characters)):
                            if ('same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][characters[i]] and value[i] != value[0]):
                                feature_instance = {}
                                break
                            feature_instance[characters[i]] = {}
                            feature_instance[characters[i]]['species'] = 'human'
                            feature_instance[characters[i]][feature] = value[i]
                            value_name += f'{value[i]}_'
                            if i == 0:
                                feature_instance[characters[i]]['color'] = gender_color_value[0][0]
                                feature_instance[characters[i]]['gender'] = gender_color_value[0][1]
                            else:
                                feature_instance[characters[i]]['color'] = gender_color_value[1][0]
                                feature_instance[characters[i]]['gender'] = gender_color_value[1][1]
                        if len(feature_instance) > 0:
                            if value_name in feature_samples[feature]:
                                feature_samples[feature][value_name] += [feature_instance]
                            else:
                                feature_samples[feature][value_name] = [feature_instance]

            else:
                if len(characters) == 1:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[0]][feature]))
                    color_values = list(product(available_values[characters[0]]['color'], available_values[characters[0]]['color']))
                elif len(characters) == 2:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[1]][feature]))
                    color_values = list(product(available_values[characters[0]]['color'], available_values[characters[1]]['color']))
                else:
                    values = list(product(available_values[characters[0]][feature], available_values[characters[1]][feature], available_values[characters[2]][feature]))
                    color_values = list(product(available_values[characters[0]]['color'], available_values[characters[1]]['color']))

                feature_samples[feature] = {}
                for value in values:
                    if feature in {'wealth', 'fitness'}:
                        if len(value) == 1:
                            if value[0] == 'normal':
                                continue
                        elif len(value) == 2:
                            if value[0] == 'normal' and value[1] == 'normal':
                                continue
                        else:
                            if value[1] == 'normal' and value[2] == 'normal':
                                continue
                    for color_value in color_values:
                        value_name = ""
                        feature_instance = {}
                        for i in range(len(characters)):
                            if ('same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][characters[i]] and value[i] != value[0]):
                                feature_instance = {}
                                break
                            feature_instance[characters[i]] = {}
                            feature_instance[characters[i]]['species'] = 'human'
                            feature_instance[characters[i]][feature] = value[i]
                            value_name += f'{value[i]}_'
                            if i == 0:
                                feature_instance[characters[i]]['color'] = color_value[0]
                            else:
                                feature_instance[characters[i]]['color'] = color_value[1]
                        if len(feature_instance) > 0:
                            if value_name in feature_samples[feature]:
                                feature_samples[feature][value_name] += [feature_instance]
                            else:
                                feature_samples[feature][value_name] = [feature_instance]

        if len(feature_samples[feature]) == 0:
            feature_samples.pop(feature)

    return feature_samples

def process_instance(dimension, dilemma, dilemma_instance, seed, iter=0):

    random.seed(seed + iter * 184)

    print(f"\nProcessing dilemma: {dilemma}...")

    if 'scene' not in DILEMMA[dimension][dilemma]:
        print(f"  Skipping '{dilemma}': no scene specified.")
        return
    
    scene = DILEMMA[dimension][dilemma]['scene']

    tmx_files = [f for f in os.listdir(f"draw_map/maps/{scene}") if f.endswith(".tmx")]
    if not tmx_files:
        print(f"  Error: TMX file not found for scene '{scene}'. Skipping.")
        return

    feature_samples = generate_single_feature_samples(dimension, dilemma, dilemma_instance)
    os.makedirs(f"{ROOT}/../data/single_feature/samples/{dimension}/{dilemma}/{dilemma_instance}", exist_ok=True)
    os.makedirs(f"{ROOT}/../data/single_feature/samples_no_desc/{dimension}/{dilemma}/{dilemma_instance}", exist_ok=True)
    features = {}
    for feature in feature_samples:
        features[feature] = []
        for value_name in feature_samples[feature]:
            feature_instance = feature_samples[feature][value_name][0]
            instance = {}
            for character in feature_instance:
                instance[character] = feature_instance[character][feature]
            features[feature].append(instance)
    with open(f"{ROOT}/../data/single_feature/samples/{dimension}/{dilemma}/{dilemma_instance}/features.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(features, f, allow_unicode=True)
    for feature in tqdm.tqdm(feature_samples):
        save_dir = f"{ROOT}/../data/single_feature/samples/{dimension}/{dilemma}/{dilemma_instance}/{feature}"
        save_dir_no_desc = f"{ROOT}/../data/single_feature/samples_no_desc/{dimension}/{dilemma}/{dilemma_instance}/{feature}"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_no_desc, exist_ok=True)

        for value in tqdm.tqdm(feature_samples[feature]):
            features = random.choice(feature_samples[feature][value])

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
            characters_by_place = {}
            sample_positions(scene, dimension, dilemma, dilemma_instance, character_description, character_quantities, characters_by_place)
            place_characters(scene, dimension, dilemma, dilemma_instance, character_description, characters_by_place, [feature], features, canvas)

            description = canvas.load_description(DILEMMA[dimension][dilemma][dilemma_instance]['description'], character_description, rewrite=True)
            canvas.render(f"{save_dir_no_desc}/{value}{iter}.jpg", load_description=False)
            canvas.render(f"{save_dir}/{value}{iter}.jpg", load_description=True)
            character_description['description'] = description
            with open(f"{save_dir}/{value}{iter}.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(character_description, f, allow_unicode=True)
            print(f"  Generated image: {save_dir}/{value}{iter}.jpg")

def main():
    parser = argparse.ArgumentParser(description="Generate dilemma image data for a specific dilemma instance.")
    parser.add_argument('--dimension', type=str, required=True, help='The dilemma dimension (e.g., care-care).')
    parser.add_argument('--dilemma', type=str, required=True, help='The specific dilemma name (e.g., trolley).')
    parser.add_argument('--dilemma-instance', type=str, required=True, help='The specific dilemma instance (e.g., trolley_0_0_0).')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iter', type=int, required=True)

    args = parser.parse_args()
    
    print(f"--- Starting task for: {args.dimension} / {args.dilemma} / {args.dilemma_instance} / {args.iter} ---")
    process_instance(args.dimension, args.dilemma, args.dilemma_instance, args.seed, args.iter)
    print(f"--- Finished task for: {args.dimension} / {args.dilemma} / {args.dilemma_instance} / {args.iter} ---")
    
if __name__ == '__main__':
    main()