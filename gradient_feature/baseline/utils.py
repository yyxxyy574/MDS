import os
import re
import random

def parse_value(value):
    return value.split('_')

def parse_description(description):
    modified_description = ""
    after_arrow = False
    parts = re.split(r'(\(\|\|ARROW:.*?\|\|)', description)
    for part in parts:
        if not part:
            continue

        if part.startswith("(||ARROW:"):
            after_arrow = True
            continue

        if after_arrow:
            if part.startswith(","):
                modified_description += f"({part.split(',', 1)[1].lstrip()}"
                continue
            elif part.startswith(")"):
                modified_description += f"{part.split(')', 1)[1].lstrip()}"
                continue

        modified_description += part
    return modified_description

def prepare_data(dataset_path, sample_ratio=None, seed=None):
    data = {}
    samples_root = os.path.join(dataset_path) 
    
    if not os.path.isdir(samples_root):
        samples_root = os.path.join(dataset_path, "samples")
    
    if not os.path.isdir(samples_root):
        return data

    for dilemma in sorted(os.listdir(samples_root)):
        dilemma_path = os.path.join(samples_root, dilemma)
        if not os.path.isdir(dilemma_path):
            continue
        
        data[dilemma] = []
        
        for subfolder in sorted(os.listdir(dilemma_path)):
            subfolder_path = os.path.join(dilemma_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for dilemma_instance in sorted(os.listdir(subfolder_path)):
                instance_path = os.path.join(subfolder_path, dilemma_instance)
                if not os.path.isdir(instance_path):
                    continue
                
                prefix = subfolder + "_"
                if not dilemma_instance.startswith(prefix):
                    continue
                
                for feature in sorted(os.listdir(instance_path)):
                    feature_path = os.path.join(instance_path, feature)
                    if not os.path.isdir(feature_path):
                        continue
                    
                    for file in sorted(os.listdir(feature_path)):
                        if not file.endswith('.yaml'):
                            continue
                        
                        yaml_path = os.path.join(feature_path, file)
                        jpg_file = file.replace('.yaml', '.jpg')
                        jpg_path = os.path.join(feature_path, jpg_file)
                        
                        if os.path.exists(jpg_path):
                            value = file.replace('.yaml', '')
                            data[dilemma].append({
                                'dimension': dilemma, 
                                'dilemma': dilemma,
                                'subfolder': subfolder,
                                'dilemma_instance': dilemma_instance,
                                'feature': feature,
                                'value': value,
                                'yaml_path': yaml_path,
                                'jpg_path': jpg_path,
                                'filename': value 
                            })

    # Sampling
    if sample_ratio is not None and sample_ratio < 1.0:
        rng = random.Random(seed)
        for dilemma in data:
            rng.shuffle(data[dilemma])
            keep = max(1, int(len(data[dilemma]) * sample_ratio))
            data[dilemma] = data[dilemma][:keep]

    return data