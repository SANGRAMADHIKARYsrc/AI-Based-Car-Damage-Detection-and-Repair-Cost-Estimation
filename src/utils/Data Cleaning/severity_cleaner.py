"""Car Damage Severity Dataset (data3a) Cleaner & Indexer.

Cleans and prepares damage severity metadata:
- Verifies image files across class subfolders (01-minor, 02-moderate, 03-severe).
- Drops corrupted or unreadable images.
- Enforces normalized integer class IDs: minor -> 0, moderate -> 1, severe -> 2.
- Generates reproducible, stratified metadata index CSV with relative filepaths.
"""

from pathlib import Path
import pandas as pd
from PIL import Image

SEVERITY_CLASSES = {
    "01-minor": 0,
    "02-moderate": 1,
    "03-severe": 2,
}


def clean_and_index_severity_dataset(data3a_dir: Path, repo_root: Path, output_dir: Path) -> dict:
    """Clean severity images and construct a standardized tabular split dataset.

    Args:
        data3a_dir: Directory containing training/ and validation/ folders.
        repo_root: Project repository root for calculating relative filepaths.
        output_dir: Path to write severity_metadata.csv.

    Returns:
        dict: Summary of cleaned records per split and class.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    corrupted_files = []

    splits = [("training", "train"), ("validation", "val")]

    for folder_name, split_tag in splits:
        split_path = data3a_dir / folder_name
        if not split_path.exists():
            continue

        for cls_folder, cls_id in SEVERITY_CLASSES.items():
            class_dir = split_path / cls_folder
            if not class_dir.exists():
                continue

            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                try:
                    with Image.open(img_file) as img:
                        img.verify()
                        width, height = img.size
                except Exception as e:
                    corrupted_files.append({"file": str(img_file), "error": str(e)})
                    continue

                rel_path = str(img_file.relative_to(repo_root)).replace("\\", "/")

                rows.append({
                    "image_id": img_file.stem,
                    "file_name": img_file.name,
                    "relative_path": rel_path,
                    "split": split_tag,
                    "severity_label": cls_folder.split("-")[1],
                    "severity_class_id": cls_id,
                    "width": width,
                    "height": height,
                })

    df = pd.DataFrame(rows)
    out_csv = output_dir / "severity_metadata.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "total_valid_images": len(df),
        "corrupted_images_dropped": len(corrupted_files),
        "split_counts": df["split"].value_counts().to_dict() if not df.empty else {},
        "class_counts": df["severity_label"].value_counts().to_dict() if not df.empty else {},
        "output_csv": str(out_csv),
    }

    return summary


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    d3a = repo_root / "data" / "Datasets" / "data3a"
    out_p = repo_root / "data" / "processed"
    if d3a.exists():
        res = clean_and_index_severity_dataset(d3a, repo_root, out_p)
        print("Severity cleaning complete:", res)
    else:
        print("data3a directory not found at:", d3a)
