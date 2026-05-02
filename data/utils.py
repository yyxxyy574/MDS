import random

from config.constants import CHARACTER, DILEMMA, SCENE
from env.gui.character import Character, POSE_DICT, ORI_DICT, GENDER_DICT, COLOR_DICT, PROFESSION_DICT, ANIMAL_DICT, UI_DICT

def sample_character(character, character_description, dimension, dilemma, dilemma_instance, feature, features):
    characteristics = {}
    if character['type'] not in character_description or character['can_describe']:
        character_description[character['type']] = ""

    if character['type'] in features:
        if 'species' in features[character['type']]:
            species_value = features[character['type']]['species']
        else:
            species_value = 'human'
        if species_value == 'non-human':
            if 'species' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']] and isinstance(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['species'], list):
                characteristics['species'] = random.choice(set(CHARACTER['species'][species_value]) & set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['species']))
            else:
                characteristics['species'] = random.choice(CHARACTER['species'][species_value])
        else:
            characteristics['species'] = 'human'
    else:
        if 'species' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
            characteristics['species'] = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['species'])
        else:
            characteristics['species'] = random.choice(['human'] + list(ANIMAL_DICT.keys()))
    
    character_description[f"{character['type']}_species"] = characteristics['species']

    if characteristics['species'] == 'human':
        
        if character['type'] in features:
            if 'color' in features[character['type']]:
                characteristics['color'] = features[character['type']]['color']
            else:
                characteristics['color'] = random.choice(CHARACTER['color'])

            if 'profession' in features[character['type']]:
                profession_value = features[character['type']]['profession']
            else:
                profession_value = None
            if profession_value is not None:
                if 'profession' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']] and isinstance(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['profession'], list):
                    characteristics['profession'] = random.choice(list(set(CHARACTER['profession'][profession_value]) & set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['profession'])))
                else:
                    characteristics['profession'] = random.choice(CHARACTER['profession'][profession_value])
            else:
                characteristics['profession'] = 'none'
            
            if 'gender' in features[character['type']]:
                characteristics['gender'] = features[character['type']]['gender']
            else:
                characteristics['gender'] = 'none'
                characteristics['profession'] = 'none'

            if 'age' in features[character['type']]:
                characteristics['age'] = features[character['type']]['age']
                if characteristics['age'] in {'infant', 'child'}:
                    characteristics['profession'] = 'child'
                elif characteristics['age'] == 'teenager':
                    characteristics['profession'] = 'student'

            if 'wealth' in features[character['type']]:
                characteristics['wealth'] = features[character['type']]['wealth']
            
            if 'fitness' in features[character['type']]:
                fitness_value = features[character['type']]['fitness']
            else:
                fitness_value = None
            if fitness_value is not None:
                if 'fitness' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']] and isinstance(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['fitness'], list):
                    characteristics['fitness'] = random.choice(list(set(CHARACTER['fitness'][fitness_value]) & set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['fitness'])))
                else:
                    characteristics['fitness'] = random.choice(CHARACTER['fitness'][fitness_value])

            if 'education' in features[character['type']]:
                characteristics['education'] = features[character['type']]['education']
            
            if 'relationship' in features[character['type']]:
                characteristics['relationship'] = features[character['type']]['relationship']

            if 'kinship' in features[character['type']]:
                kinship_value = features[character['type']]['kinship']
            else:
                kinship_value = None
            if kinship_value is not None:
                if 'kinship' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']] and isinstance(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['kinship'], list):
                    characteristics['kinship'] = random.choice(list(set(CHARACTER['kinship'][kinship_value]) & set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['kinship'])))
                else:
                    characteristics['kinship'] = random.choice(CHARACTER['kinship'][kinship_value])
            
            # character_description[character['type']] += " "
            for feature_dimension in feature:
                if feature_dimension != 'species' and characteristics[feature_dimension] not in {'none', 'normal'}:
                    character_description[character['type']] += f"{characteristics[feature_dimension]} "
            if 'profession' in feature:
                character_description[character['type']] += ", "
            elif 'age' in feature or 'gender' in feature:
                if 'profession' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']] and 'none' not in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['profession']:
                    profession = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['profession'])
                    character_description[character['type']] += f"{profession}, "
                else:
                    character_description[character['type']] += f", "
            else:
                if 'profession' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']] and 'none' not in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['profession']:
                    profession = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['profession'])
                    character_description[character['type']] += f"{profession}, "
                else:
                    character_description[character['type']] += f"{characteristics['species']}, "
        else:
            if 'gender' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                characteristics['gender'] = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['gender'])
            else:
                characteristics['gender'] = random.choice(list(GENDER_DICT.keys()))

            if 'color' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                characteristics['color'] = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['color'])
            else:
                characteristics['color'] = random.choice(list(COLOR_DICT.keys()))

            if 'profession' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                characteristics['profession'] = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['profession'])
            else:
                if 'age' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                    possible_professions = []
                    if set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['age']) & {'infant', 'child'}:
                        possible_professions += ['child']
                    if set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['age']) & {'child', 'teenager'}:
                        possible_professions += ['student']
                    if set(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['age']) & {'middle-age', 'elderly'}:
                        possible_professions += ['thief', 'blue-collar', 'chef', 'unemployed', 'police', 'doctor', 'teacher', 'white-collar', 'boss', 'soldier', 'artist']
                    characteristics['profession'] = random.choice(possible_professions)
                else:
                    characteristics['profession'] = random.choice(list(PROFESSION_DICT.keys()))
    else:
        characteristics['gender'] = ''
        characteristics['color'] = ''
        characteristics['profession'] = ''
        character_description[character['type']] += f"{characteristics['species']}, "

    return characteristics['species'], characteristics['gender'], characteristics['color'], characteristics['profession']

def sample_positions(scene, dimension, dilemma, dilemma_instance, character_description, character_quantities, characters_by_place):
    characters_to_instantiate = []
    priority_characters = []
    flexible_characters = []

    for character in DILEMMA[dimension][dilemma][dilemma_instance]['character']:

        quantity = character_quantities[character]
        
        if quantity == 0:
            character_description[character] = ""
            continue

        if DILEMMA[dimension][dilemma][dilemma_instance]['character'][character] and  'place' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:

            is_ui_load = [False] * quantity
            ui = None
            ui_pos = None
            if 'ui' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                ui = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['ui'])
                if ui.split('_box')[0] in UI_DICT:
                    ui_pos = random.choice(UI_DICT[ui.split('_box')[0]])
                else:
                    ui_pos = None
                if quantity > 2:
                    ui_load_indices = random.sample(range(quantity), random.randint(1, min(5, quantity // 2 + 1)))
                else:
                    ui_load_indices = random.sample(range(quantity), 1)
                for j in ui_load_indices:
                    is_ui_load[j] = True

            for j in range(quantity):
                # place = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place'])
                character_inst = {
                    'type': character,
                    # 'place': place,
                    'places': DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place'].copy(),
                    'is_ui_load': is_ui_load[j],
                    'ui': ui,
                    'ui_pos': ui_pos,
                    'is_text': 'text' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character],
                    'can_describe': 'description' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place']
                }
                characters_to_instantiate.append(character_inst)
                if 'description' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place']:
                    flexible_characters.append(character_inst)
                else:
                    priority_characters.append(character_inst)

        if DILEMMA[dimension][dilemma][dilemma_instance]['character'][character] and  'arrow' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
            character_description[f'{character}_color'] = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['arrow']

    available_positions = {}
    for place_name in SCENE[scene]:
        if 'pos' in SCENE[scene][place_name]:
            available_positions[place_name] = SCENE[scene][place_name]['pos'].copy()
        elif 'scope' in SCENE[scene][place_name]:
            available_positions[place_name] = ['scope']
    
    successfully_placed_characters = []
    failed_characters = []

    if len(priority_characters) >0 and priority_characters[0]['type'] == 'agent':
        sublist = priority_characters[1:]
        random.shuffle(sublist)
        priority_characters[1:] = sublist
    else:
        random.shuffle(priority_characters)
    random.shuffle(flexible_characters)
    for character in priority_characters + flexible_characters:
        placed = False
        available_places = [p for p in character['places'] if p != 'description']
        random.shuffle(available_places)

        for place in available_places:
            if place in available_positions and available_positions[place]:
                if available_positions[place] == ['scope']:
                    character['assigned_place'] = place
                    successfully_placed_characters.append(character)
                    placed = True
                    break
                elif len(available_positions[place]) > 0:
                    pos = random.choice(available_positions[place])
                    available_positions[place].remove(pos)
                    character['assigned_place'] = place
                    character['assigned_pos'] = pos
                    successfully_placed_characters.append(character)
                    placed = True
                    break
            
        if not placed:
            failed_characters.append(character)

    fail = False
    for character in failed_characters:
        if character['can_describe']:
            character['assigned_place'] = 'description'
            successfully_placed_characters.append(character)
        else:
            fail = True
            print(f"  Warning: Could not place character of type '{character['type']}'")

    for character in successfully_placed_characters:
        place = character['assigned_place']
        if place not in characters_by_place:
            characters_by_place[place] = []
        characters_by_place[place].append(character)

    return fail

def place_characters(scene, dimension, dilemma, dilemma_instance, character_description, characters_by_place, feature, features, canvas):
    for place, characters in characters_by_place.items():

        if place == 'description':
            for character in characters:
                sample_character(character, character_description, dimension, dilemma, dilemma_instance, feature, features)
            continue

        if scene not in SCENE or place not in SCENE[scene]:
            print(f"  Warning: Place '{place}' not found in scene '{scene}' config.")
            continue

        if characters[0]['is_text']:
            character = characters[0]
            template = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['text']
            scope = SCENE[scene][place]['scope']
            layer = SCENE[scene][place]['layer']

            species, gender, color, profession = sample_character(character, character_description, dimension, dilemma, dilemma_instance, feature, features)

            if species == 'human':
                pose = random.choice(list(POSE_DICT.keys()))
                if 'pose' in SCENE[scene][character['assigned_place']]:
                    pose = random.choice(SCENE[scene][character['assigned_place']]['pose'])
                if 'pose' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                    pose = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['pose'])
                
                if 'ori' in SCENE[scene][character['assigned_place']]:
                    ori = random.choice(SCENE[scene][character['assigned_place']]['ori'])
                else:
                    ori = 'up'

                role = Character(x, y, canvas.tile_w, canvas.tile_h, ori=ori, pose=pose, profession=profession, gender=gender, color=color, arrow=arrow)
            else:
                role = Character(x, y, canvas.tile_w, canvas.tile_h, species=species, arrow=arrow)

            bottom_pos = canvas.load_text(template, character_description, scope, layer)

            if 'ui' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                ui = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['ui'])
                if ui.split('_box')[0] in UI_DICT:
                    ui_pos = random.choice(UI_DICT[ui.split('_box')[0]])
                else:
                    ui_pos = None
                if ui_pos is None:
                    ui_pos = [role.pose_x, role.pose_y]
                    canvas.load_ui(ui_pos, bottom_pos, ui.endswith('_box'), species == 'human', species in ANIMAL_DICT)
                else:
                    canvas.load_ui(ui_pos, bottom_pos, ui.endswith('_box'))
        else:
            assigned_positions = []
            for character in characters:
                if 'assigned_pos' in character:
                    assigned_positions.append((character, character['assigned_pos']))

            assigned_positions.sort(key=lambda x: x[1][1])

            for character, (x, y) in assigned_positions:

                arrow = None
                if 'arrow' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                    arrow = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['arrow']

                species, gender, color, profession = sample_character(character, character_description, dimension, dilemma, dilemma_instance, feature, features)

                if species == 'human':
                    pose = random.choice(list(POSE_DICT.keys()))
                    if 'pose' in SCENE[scene][character['assigned_place']]:
                        pose = random.choice(SCENE[scene][character['assigned_place']]['pose'])
                    if 'pose' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]:
                        pose = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['pose'])
                    
                    if 'ori' in SCENE[scene][character['assigned_place']]:
                        ori = random.choice(SCENE[scene][character['assigned_place']]['ori'])
                    else:
                        ori = 'up'

                    role = Character(x, y, canvas.tile_w, canvas.tile_h, ori=ori, pose=pose, profession=profession, gender=gender, color=color, arrow=arrow)

                    canvas.load_character(role, layer=SCENE[scene][place]['layer'])
                
                else:
                    role = Character(x, y, canvas.tile_w, canvas.tile_h, species=species, arrow=arrow)

                    canvas.load_character(role, layer=SCENE[scene][place]['layer'])

                if character['is_ui_load']:
                    ui = character['ui']
                    is_box = ui.endswith('_box')
                    ui_pos = character['ui_pos']
                    if 'is_ui_same' not in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']] or not DILEMMA[dimension][dilemma][dilemma_instance]['character'][character['type']]['is_ui_same']:
                        if ui.split('_box')[0] in UI_DICT:
                            ui_pos = random.choice(UI_DICT[ui.split('_box')[0]])
                        else:
                            ui_pos = None
                    
                    if is_box:
                        top_pos = [x  * canvas.tile_w + canvas.tile_w // 2, (y - 2) * canvas.tile_h]
                    else:
                        top_pos = [x  * canvas.tile_w + canvas.tile_w // 2, (y - 1) * canvas.tile_h]

                    if ui_pos is None:
                        ui_pos = [role.pose_x, role.pose_y]
                        canvas.load_ui(ui_pos, top_pos, is_box, species == 'human', species in ANIMAL_DICT)
                    else:
                        canvas.load_ui(ui_pos, top_pos, is_box)

def sample_quantities(scene, dimension, dilemma, dilemma_instance, character_description, character_quantities):
    for character in DILEMMA[dimension][dilemma][dilemma_instance]['character']:
        if DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
            if 'quantity' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                quantity = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['quantity'])
            elif 'min_quantity' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                if 'description' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place'] or 'text' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                    quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity'], DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'])
                else:
                    bias = 0
                    if 'quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                        bias = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['quantity_bias']
                    pos_quantity = 0
                    for place in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place']:
                        if place in SCENE[scene]:
                            pos_quantity += len(SCENE[scene][place]['pos'])
                    if min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'], pos_quantity) + bias < DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity']:
                        quantity = min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'], pos_quantity) + bias
                    else:
                        quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity'], min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'], pos_quantity) + bias)
            elif 'range' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['range'][0], DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['range'][1])
            else:
                quantity = random.randint(0, 100)
        else:
            quantity = 0

        character_quantities[character] = quantity
        character_description[f'{character}_quantity'] = quantity

def sample_same_quantities(scene, dimension, dilemma, dilemma_instance, character_description, character_quantities):
    same_quantity = None
    for character in DILEMMA[dimension][dilemma][dilemma_instance]['character']:
        if DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
            if 'same_quantity' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                same_bias = 0
                if 'same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                    same_bias = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity_bias']
                if same_quantity is None:
                    if 'description' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place'] or 'text' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                        quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity'][0], DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity'][1])
                    else:
                        bias = 0
                        if 'quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                            bias = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['quantity_bias']
                        pos_quantity = 0
                        for place in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place']:
                            if place in SCENE[scene]:
                                pos_quantity += len(SCENE[scene][place]['pos'])
                        if min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity'][1], pos_quantity) + bias < DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity'][0]:
                            quantity = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity'][0]
                        else:
                            quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity'][0], min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity'][1], pos_quantity) + bias)
                    same_quantity = quantity - same_bias
                else:
                    if 'description' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place'] or 'text' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                        quantity = same_quantity + same_bias
                    else:
                        bias = 0
                        if 'quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                            bias = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['quantity_bias']
                        pos_quantity = 0
                        for place in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place']:
                            if place in SCENE[scene]:
                                pos_quantity += len(SCENE[scene][place]['pos'])
                        while(same_quantity != DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity'][0]):
                            if same_quantity <= pos_quantity + bias:
                                break
                            same_quantity -= 1
                        quantity = same_quantity + same_bias
                        
            elif 'quantity' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                quantity = random.choice(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['quantity'])
            
            elif 'min_quantity' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                if 'description' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place'] or 'text' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                    quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity'], DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'])
                else:
                    bias = 0
                    if 'quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                        bias = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['quantity_bias']
                    pos_quantity = 0
                    for place in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['place']:
                        if place in SCENE[scene]:
                            pos_quantity += len(SCENE[scene][place]['pos'])
                    if min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'], pos_quantity) + bias < DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity']:
                        # quantity = min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'], pos_quantity) + bias
                        quantity = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity']
                    else:
                        quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['min_quantity'], min(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['max_quantity'], pos_quantity) + bias)

            elif 'range' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                quantity = random.randint(DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['range'][0], DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['range'][1])
            else:
                quantity = random.randint(0, 100)
        else:
            quantity = 0

        character_quantities[character] = quantity
        character_description[f'{character}_quantity'] = quantity
    
    for character in DILEMMA[dimension][dilemma][dilemma_instance]['character']:
        if DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
            same_bias = 0
            if 'same_quantity_bias' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]:
                same_bias = DILEMMA[dimension][dilemma][dilemma_instance]['character'][character]['same_quantity_bias']
            if 'same_quantity' in DILEMMA[dimension][dilemma][dilemma_instance]['character'][character] and same_quantity is not None:
                character_quantities[character] = same_quantity + same_bias
                character_description[f'{character}_quantity'] = same_quantity +same_bias