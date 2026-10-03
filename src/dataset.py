"""
Dataset Ingestion, Verification, and Manifest Generator for CHASE_DB1 Retinal Database.
Ensures patient-level split (Subjects 01-10 Train, Subjects 11-14 Test) to prevent data leakage.
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "new" / "chase"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def build_dataset_manifest(data_dir: Path = DATA_DIR, save_csv: bool = True) -> pd.DataFrame:
    """
    Scans the CHASE_DB1 directory structure and builds an aligned, verified manifest DataFrame.
    """
    records = []
    
    # 1. Process Training Set (20 images: Subjects 01L to 10R)
    train_dir = data_dir / "training" / "training"
    train_images = sorted(os.listdir(train_dir / "images"))
    train_m1 = sorted(os.listdir(train_dir / "1st_manual"))
    train_m2 = sorted(os.listdir(train_dir / "2nd_manual"))
    train_masks = sorted(os.listdir(train_dir / "mask"))
    
    for idx in range(len(train_images)):
        # Decode subject and eye side from 2nd manual filename (e.g. Image_01L_2ndHO.png)
        m2_name = train_m2[idx]
        base_id = m2_name.replace("_2ndHO.png", "")  # e.g., Image_01L
        subject_id = base_id.split("_")[1][:2]       # e.g., 01
        eye_side = base_id.split("_")[1][2]         # e.g., L
        
        img_path = train_dir / "images" / train_images[idx]
        m1_path = train_dir / "1st_manual" / train_m1[idx]
        m2_path = train_dir / "2nd_manual" / train_m2[idx]
        mask_path = train_dir / "mask" / train_masks[idx]
        
        # Verify file existence and readability
        assert img_path.exists(), f"Image missing: {img_path}"
        assert m1_path.exists(), f"1st manual missing: {m1_path}"
        assert m2_path.exists(), f"2nd manual missing: {m2_path}"
        assert mask_path.exists(), f"FOV mask missing: {mask_path}"
        
        records.append({
            "image_id": base_id,
            "patient_id": f"Subject_{subject_id}",
            "eye": eye_side,
            "split": "train",
            "image_filename": train_images[idx],
            "image_path": str(img_path.resolve()),
            "manual1_path": str(m1_path.resolve()),
            "manual2_path": str(m2_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "width": 999,
            "height": 960
        })
        
    # 2. Process Testing Set (8 images: Subjects 11L to 14R)
    test_dir = data_dir / "test" / "test"
    test_images = sorted(os.listdir(test_dir / "images"))
    test_m1 = sorted(os.listdir(test_dir / "1st_manual"))
    test_m2 = sorted(os.listdir(test_dir / "2nd_manual"))
    test_masks = sorted(os.listdir(test_dir / "mask"))
    
    for idx in range(len(test_images)):
        m2_name = test_m2[idx]
        base_id = m2_name.replace("_2ndHO.png", "")  # e.g., Image_11L
        subject_id = base_id.split("_")[1][:2]       # e.g., 11
        eye_side = base_id.split("_")[1][2]         # e.g., L
        
        img_path = test_dir / "images" / test_images[idx]
        m1_path = test_dir / "1st_manual" / test_m1[idx]
        m2_path = test_dir / "2nd_manual" / test_m2[idx]
        mask_path = test_dir / "mask" / test_masks[idx]
        
        assert img_path.exists(), f"Test image missing: {img_path}"
        assert m1_path.exists(), f"Test 1st manual missing: {m1_path}"
        assert m2_path.exists(), f"Test 2nd manual missing: {m2_path}"
        assert mask_path.exists(), f"Test FOV mask missing: {mask_path}"
        
        records.append({
            "image_id": base_id,
            "patient_id": f"Subject_{subject_id}",
            "eye": eye_side,
            "split": "test",
            "image_filename": test_images[idx],
            "image_path": str(img_path.resolve()),
            "manual1_path": str(m1_path.resolve()),
            "manual2_path": str(m2_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "width": 999,
            "height": 960
        })
        
    df = pd.DataFrame(records)
    
    if save_csv:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = OUTPUTS_DIR / "dataset_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"[Dataset] Manifest generated with {len(df)} images ({len(df[df.split=='train'])} train, {len(df[df.split=='test'])} test) -> {manifest_path}")
        
    return df


if __name__ == "__main__":
    df = build_dataset_manifest()
    print(df[["image_id", "patient_id", "eye", "split"]].head(10))
