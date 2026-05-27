import argparse
import zipfile
import json
from pathlib import Path
import pandas as pd
from collections import Counter

def load_posts(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = df.rename(columns={
        "Id": "post_id",
        "CategoryId": "category_id",
        "categoryname": "category",
        "Text": "text",
    })
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 5].copy()
    return df

def load_photos(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = df.rename(columns={
        "Id": "photo_id",
        "Url": "url",
        "DataSource": "data_source",
        "PostId": "post_id",
    })
    df = df[df["url"].notna() & df["url"].str.startswith("http")].copy()
    return df

def extract_subcategory_images(zip_path: str, output_dir: str) -> dict:
    from taxonomy import SUBCATEGORY_FOLDER_MAP
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_meta = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            folder_name = None
            for part in parts:
                if part in SUBCATEGORY_FOLDER_MAP:
                    folder_name = part
                    break
            if folder_name is None:
                continue
            ext = Path(name).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            target_path = output_dir / name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            subcat = SUBCATEGORY_FOLDER_MAP[folder_name]
            image_meta[str(target_path)] = {
                "subcategory": subcat,
                "category": "Страйкбольное оружие",
            }
    meta_path = output_dir / "image_labels.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(image_meta, f, ensure_ascii=False, indent=2)
    return image_meta

def merge_posts_photos(posts_df: pd.DataFrame, photos_df: pd.DataFrame) -> pd.DataFrame:
    photo_groups = (
        photos_df.groupby("post_id")["url"]
        .apply(list)
        .reset_index()
        .rename(columns={"url": "photo_urls"})
    )
    photo_id_groups = (
        photos_df.groupby("post_id")["photo_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"photo_id": "photo_ids"})
    )
    photo_groups = photo_groups.merge(photo_id_groups, on="post_id")
    merged = posts_df.merge(photo_groups, on="post_id", how="left")
    merged["photo_urls"] = merged["photo_urls"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged["photo_ids"] = merged["photo_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged["has_photos"] = merged["photo_urls"].apply(len) > 0
    return merged

def save_processed(df: pd.DataFrame, output_path: str):
    df.to_parquet(output_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", default="posts.parquet")
    parser.add_argument("--photos", default="photos.parquet")
    parser.add_argument("--zip", default="subcategory_images.zip")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    posts_df = load_posts(args.posts)
    photos_df = load_photos(args.photos)
    merged = merge_posts_photos(posts_df, photos_df)
    save_processed(merged, str(output_dir / "dataset.parquet"))
    if Path(args.zip).exists():
        extract_subcategory_images(args.zip, str(output_dir / "images"))

if __name__ == "__main__":
    main()