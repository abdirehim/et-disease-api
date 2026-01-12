import os
import shutil
import random
import json
import argparse
import yaml
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_data(config):
    raw_path = Path(config['raw_data_path'])
    processed_path = Path(config['data_path'])
    img_size = config['img_size']
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (processed_path / split).mkdir(parents=True, exist_ok=True)
        
    # Get all classes
    classes = [d.name for d in raw_path.iterdir() if d.is_dir()]
    classes.sort()
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create labels mapping
    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    pd.DataFrame(list(label_map.items()), columns=['class_name', 'label_idx']).to_csv(processed_path / 'labels.csv', index=False)
    
    all_files = []
    
    # Collect all files
    for cls_name in classes:
        cls_dir = raw_path / cls_name
        files = list(cls_dir.glob('*'))
        files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        for f in files:
            all_files.append({'path': f, 'class': cls_name, 'label': label_map[cls_name]})
            
    # Split data
    train_files, test_files = train_test_split(all_files, test_size=0.3, stratify=[f['class'] for f in all_files], random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, stratify=[f['class'] for f in test_files], random_state=42) # 15% val, 15% test
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Process and save images
    split_info = {}
    
    for split_name, files in splits.items():
        print(f"Processing {split_name} split ({len(files)} images)...")
        split_info[split_name] = []
        
        split_dir = processed_path / split_name
        
        for item in tqdm(files):
            # Create class subdir in split
            cls_dir = split_dir / item['class']
            cls_dir.mkdir(exist_ok=True)
            
            # Read and resize
            img = cv2.imread(str(item['path']))
            if img is None:
                print(f"Warning: Could not read {item['path']}")
                continue
                
            img = cv2.resize(img, (img_size, img_size))
            
            # Save
            dest_name = item['path'].name
            dest_path = cls_dir / dest_name
            cv2.imwrite(str(dest_path), img)
            
            split_info[split_name].append({
                'filename': dest_name,
                'class': item['class'],
                'label': item['label']
            })
            
    # Save splits info
    with open(processed_path / 'splits.json', 'w') as f:
        # Convert Path objects to str first if necessary, but here we construct new dicts
        # actually split_info has strings mostly.
        json.dump(split_info, f, indent=2)
        
    print("Data preprocessing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Create dummy raw data if it doesn't exist for demonstration/init purposes?
    # No, assuming raw data exists or user puts it there. 
    # But ensuring directories exist for now.
    
    config = load_config(args.config)
    
    # Check if raw data exists, otherwise warn
    if not os.path.exists(config['raw_data_path']):
        print(f"Error: Raw data path {config['raw_data_path']} does not exist.")
        # Create it just so script doesn't crash on empty run, or instructions?
        # User instructions implied reading from there. I'll leave it to fail if missing, 
        # as user likely needs to provide data.
        os.makedirs(config['raw_data_path'], exist_ok=True) 
        print(f"Created empty {config['raw_data_path']}. Please place images there.")
    else:
        process_data(config)
