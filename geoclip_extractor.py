#!/usr/bin/env python3
"""
GeoCLIP Incremental Data Extractor
- Scans for images with GPS data
- Builds and maintains a master index
- Incrementally uploads new data to HuggingFace
"""

import os
import hashlib
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
from tqdm.notebook import tqdm

# Optional HuggingFace support
try:
    from datasets import Dataset, Features, Value, Image as HFImage
    from huggingface_hub import login, whoami
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configuration (customize these)
ROOT = Path("data")                     # Root directory for images
MASTER = Path("metadata/master.csv")    # Master index file
HF_REPO = "your-username/geoclip-dataset"  # HuggingFace repo name

def sha256(path):
    """Generate hash for a file to track changes"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_image_gps_metadata(image_path):
    """Extract GPS metadata from an image file"""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            return None, None
        
        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                gps_info = value
                break

        if not gps_info:
            return None, None

        # Extract latitude and longitude
        lat_deg, lat_min, lat_sec = gps_info.get(2, (0, 0, 0))
        lon_deg, lon_min, lon_sec = gps_info.get(4, (0, 0, 0))
        lat_ref, lon_ref = gps_info.get(1, 'N'), gps_info.get(3, 'E')

        latitude = lat_deg + (lat_min / 60.0) + (lat_sec / 3600.0)
        longitude = lon_deg + (lon_min / 60.0) + (lon_sec / 3600.0)

        if lat_ref == 'S':
            latitude = -latitude
        if lon_ref == 'W':
            longitude = -longitude

        return latitude, longitude
    except Exception as e:
        return None, None

def load_master():
    """Load the master index or create a new one if it doesn't exist"""
    return pd.read_csv(MASTER) if MASTER.exists() else pd.DataFrame(columns=["image", "lat", "lon", "hash"])

def save_master(df):
    """Save the master index"""
    MASTER.parent.mkdir(exist_ok=True)
    df.to_csv(MASTER, index=False)
    
    # Also save a GeoCLIP-formatted version for training
    geoclip_df = df[["image", "lat", "lon"]].copy()
    geoclip_df.columns = ["IMG_FILE", "LAT", "LON"]
    geoclip_df.to_csv(MASTER.parent / "geoclip_training.csv", index=False)
    return MASTER.parent / "geoclip_training.csv"

def scan_images():
    """Find all image files in the root directory"""
    return [p for p in ROOT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

def ingest():
    """Process images and update the master index"""
    # Load existing data
    df = load_master()
    seen = set(df["hash"])
    
    # Scan for images
    all_images = scan_images()
    print(f"Found {len(all_images)} total images")
    
    # Extract GPS for new images
    new_rows = []
    
    for img in tqdm(all_images, desc="Processing images", unit="img"):
        try:
            h = sha256(img)
            if h in seen:
                continue
                
            lat, lon = get_image_gps_metadata(img)
            if lat is not None and lon is not None:
                new_rows.append({
                    "image": str(img), 
                    "lat": lat, 
                    "lon": lon, 
                    "hash": h
                })
        except Exception as e:
            print(f"Error processing {img}: {e}")
    
    # Update master index if new data found
    if not new_rows:
        geoclip_path = MASTER.parent / "geoclip_training.csv"
        return df, None, geoclip_path
        
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    geoclip_path = save_master(df)
    
    return df, new_rows, geoclip_path

def push_to_hf(new_rows, private=True):
    """Push new data to HuggingFace"""
    if not HF_AVAILABLE:
        print("❌ HuggingFace libraries not available. Install with:")
        print("!pip install datasets huggingface_hub")
        return False
    
    # Prepare dataset with only needed columns
    upload_df = pd.DataFrame(new_rows)[["image", "lat", "lon"]]
    
    # Verify all image paths exist
    missing = [p for p in upload_df["image"] if not os.path.exists(p)]
    if missing:
        print(f"❌ {len(missing)} image paths don't exist")
        return False
    
    # Create HuggingFace dataset
    ds = Dataset.from_pandas(upload_df, features=Features({
        "image": HFImage(),
        "lat": Value("float32"),
        "lon": Value("float32")
    }))
    
    # Login to HuggingFace
    try:
        try:
            whoami()
            print("Already logged in to HuggingFace Hub")
        except:
            # Try environment variable first
            token = os.getenv("HF_TOKEN")
            if not token:
                token = input("Enter your HuggingFace token: ")
            login(token=token)
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False
    
    # Upload to HuggingFace
    print(f"Uploading {len(upload_df)} new images to '{HF_REPO}'...")
    ds.push_to_hub(HF_REPO, private=private)
    
    print(f"✅ Data uploaded to: https://huggingface.co/datasets/{HF_REPO}")
    return True

def main(root_dir=None, master_file=None, hf_repo=None, upload=True, private=True):
    """Main entry point with configurable parameters"""
    # Update global configuration if specified
    global ROOT, MASTER, HF_REPO
    if root_dir:
        ROOT = Path(root_dir)
    if master_file:
        MASTER = Path(master_file)
    if hf_repo:
        HF_REPO = hf_repo
    
    # Process images
    df, new_rows, geoclip_path = ingest()
    
    # Print results
    print(f"✅ Master index now contains {len(df)} images total")
    
    if new_rows:
        print(f"➕ Added {len(new_rows)} new images with GPS data")
        print(f"📄 GeoCLIP training file saved to: {geoclip_path}")
        
        # Upload to HuggingFace if requested
        if upload:
            if HF_AVAILABLE:
                push_to_hf(new_rows, private)
            else:
                print("⚠️ HuggingFace upload skipped (libraries not available)")
    else:
        print("🔄 No new images with GPS data found")
    
    return df, new_rows, geoclip_path

if __name__ == "__main__":
    # You can customize these parameters
    main(
        root_dir="data",                     # Root directory for images
        master_file="metadata/master.csv",   # Master index file
        hf_repo="your-username/geoclip-dataset",  # HuggingFace repo name
        upload=True,                         # Whether to upload to HuggingFace
        private=True                         # Whether the HuggingFace dataset is private
    )

# Example usage in a notebook:
# from geoclip_extractor import main
# df, new_rows, geoclip_path = main(
#     root_dir="/content/drive/MyDrive/images",
#     master_file="/content/drive/MyDrive/metadata/master.csv",
#     hf_repo="your-username/geoclip-dataset"
# )


# Run with default parameters (from command line)
!python geoclip_extractor.py
#dont forget your hf hub login or key if needed 
# Or customize in a notebook
from geoclip_extractor import main
df, new_rows, geoclip_path = main(
    root_dir="/content/drive/MyDrive/theo2",
    master_file="/content/drive/MyDrive/metadata/master.csv",
    hf_repo="latterworks/geoclip-dataset"
)
