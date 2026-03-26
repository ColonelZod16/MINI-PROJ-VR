import os
import json
import shutil
import random
from tqdm import tqdm
from collections import Counter

# --- CONFIGURATION ---
SOURCE_DIR = r"train-003/train"
IMG_DIR = os.path.join(SOURCE_DIR, "image")

# prefer "annotations", fallback to "annos"
_ANN_A = os.path.join(SOURCE_DIR, "annotations")
_ANN_B = os.path.join(SOURCE_DIR, "annos")
ANN_DIR = _ANN_A if os.path.isdir(_ANN_A) else _ANN_B

OUTPUT_DIR = "Pruned_Clothing_Dataset_30k"

TOP_K = 5
TRAIN_SIZE = 30000
VAL_SIZE = 4000
TEST_SIZE = 4000
RANDOM_SEED = 42

# DeepFashion2 category map
CATEGORY_ID_TO_NAME = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}


def get_category_from_ann(ann_path):
    """
    Parse one annotation JSON and return a single category label.
    Priority:
      1) DeepFashion2 keys: item1/item2/... with category_name/category_id
      2) fallback schemas: item_category, label, annotations, shapes
    """
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # 1) DeepFashion2-style: item1, item2, ...
    item_keys = sorted([k for k in data.keys() if k.startswith("item")])
    for k in item_keys:
        item = data.get(k)
        if not isinstance(item, dict):
            continue

        # explicit name
        cat_name = item.get("category_name")
        if cat_name not in (None, ""):
            return str(cat_name)

        # id mapped to readable name
        cat_id = item.get("category_id")
        if cat_id is not None:
            try:
                cat_id = int(cat_id)
                return CATEGORY_ID_TO_NAME.get(cat_id, f"category_{cat_id}")
            except Exception:
                return str(cat_id)

    # 2) fallback schemas
    if "item_category" in data and data["item_category"] not in (None, ""):
        return str(data["item_category"])

    if "label" in data and data["label"] not in (None, ""):
        return str(data["label"])

    anns = data.get("annotations")
    if isinstance(anns, list) and len(anns) > 0 and isinstance(anns[0], dict):
        x = anns[0].get("category_name") or anns[0].get("category_id")
        if x not in (None, ""):
            return str(x)
    if isinstance(anns, dict):
        x = anns.get("category_name") or anns.get("category_id")
        if x not in (None, ""):
            return str(x)

    shapes = data.get("shapes")
    if isinstance(shapes, list) and len(shapes) > 0 and isinstance(shapes[0], dict):
        x = shapes[0].get("label")
        if x not in (None, ""):
            return str(x)
    if isinstance(shapes, dict):
        x = shapes.get("label")
        if x not in (None, ""):
            return str(x)

    return None


def prune_dataset():
    random.seed(RANDOM_SEED)

    if not os.path.exists(IMG_DIR):
        print(f"ERROR: Could not find image folder at {IMG_DIR}")
        return
    if not os.path.exists(ANN_DIR):
        print(f"ERROR: Could not find annotation folder at {ANN_DIR}")
        return

    # 1) Map all images to categories
    print(f"--- Scanning {SOURCE_DIR} for Top {TOP_K} categories ---")
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    manifest = []
    cat_counter = Counter()

    missing_ann = 0
    invalid_cat = 0

    for img_name in tqdm(image_files, desc="Parsing Annotations"):
        stem, _ = os.path.splitext(img_name)
        ann_name = f"{stem}.json"
        ann_path = os.path.join(ANN_DIR, ann_name)

        if not os.path.exists(ann_path):
            missing_ann += 1
            continue

        cat = get_category_from_ann(ann_path)
        if cat is None or str(cat).strip() == "":
            invalid_cat += 1
            continue

        cat = str(cat).strip()
        cat_counter[cat] += 1
        manifest.append((img_name, ann_name, cat))

    print(
        f"[INFO] total_images={len(image_files)} "
        f"missing_ann={missing_ann} invalid_cat={invalid_cat} valid_pairs={len(manifest)}"
    )

    if not manifest:
        print("\nERROR: No Image-Annotation pairs with valid categories were found.")
        print(f"[DEBUG] IMG_DIR={os.path.abspath(IMG_DIR)}")
        print(f"[DEBUG] ANN_DIR={os.path.abspath(ANN_DIR)}")
        return

    # 2) Select Top-K categories
    top_k_pairs = cat_counter.most_common(TOP_K)
    top_k_labels = [cat for cat, _ in top_k_pairs]
    print(f"\nTop {TOP_K} Categories Found: {top_k_pairs}")

    # Keep only top-k
    manifest = [item for item in manifest if item[2] in top_k_labels]

    # 3) Handle class imbalance via undersampling
    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    per_class_limit = max(1, total_needed // TOP_K)

    final_selection = []
    for cat_label in top_k_labels:
        cat_pool = [item for item in manifest if item[2] == cat_label]
        sample_size = min(len(cat_pool), per_class_limit)
        sampled = random.sample(cat_pool, sample_size) if sample_size > 0 else []
        final_selection.extend(sampled)
        print(f"✓ Balanced {cat_label}: selected {len(sampled)} items")

    if not final_selection:
        print("\nERROR: No samples selected after top-k filtering.")
        return

    # 4) Create splits
    random.shuffle(final_selection)

    total_available = len(final_selection)
    desired_total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if total_available < desired_total:
        print(
            f"\n[WARN] Only {total_available} samples available, "
            f"less than requested {desired_total}. Using available samples."
        )

    train_n = min(TRAIN_SIZE, total_available)
    rem = total_available - train_n
    val_n = min(VAL_SIZE, rem)
    rem -= val_n
    test_n = min(TEST_SIZE, rem)

    train_data = final_selection[:train_n]
    val_data = final_selection[train_n: train_n + val_n]
    test_data = final_selection[train_n + val_n: train_n + val_n + test_n]

    # 5) Copy files
    splits = {"train": train_data, "val": val_data, "test": test_data}

    for split_name, data_list in splits.items():
        print(f"\n--- Saving {split_name} split ({len(data_list)} images) ---")
        img_out = os.path.join(OUTPUT_DIR, split_name, "images")
        ann_out = os.path.join(OUTPUT_DIR, split_name, "annotations")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(ann_out, exist_ok=True)

        for img_name, ann_name, _cat in tqdm(data_list, desc=f"Copying {split_name}"):
            shutil.copy2(os.path.join(IMG_DIR, img_name), os.path.join(img_out, img_name))
            shutil.copy2(os.path.join(ANN_DIR, ann_name), os.path.join(ann_out, ann_name))

    print(f"\nSuccess! Pruned dataset created at: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    prune_dataset()