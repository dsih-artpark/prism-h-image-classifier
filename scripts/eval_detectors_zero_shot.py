import os
import sys
import subprocess

# --- Setup & Dependencies ---
def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "opencv-python", "matplotlib", "transformers", "scipy", "pycocotools", "ultralytics", "tqdm"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/IDEA-Research/GroundingDINO.git"])

    # Download Weights (if needed)
    if not os.path.exists("groundingdino_swint_ogc.pth"):
        subprocess.check_call(["wget", "-q", "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"])
    if not os.path.exists("groundingdino_swinb_cogcoor.pth"):
        subprocess.check_call(["wget", "-q", "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"])

# Uncomment to install deps
# install_deps()

import json
import glob
import time
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Model Imports
from transformers import Owlv2Processor, Owlv2ForObjectDetection
try:
    from groundingdino.util.inference import load_model, predict as groundingdino_predict
    import groundingdino.datasets.transforms as T
except ImportError:
    print("GroundingDINO not installed properly.")
try:
    from ultralytics import YOLOWorld
except ImportError:
    print("Ultralytics not installed.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Configuration ---
# Options: "gdino_swint", "gdino_swinb", "owlv2", "yolo_world"
MODEL_BACKEND = "owlv2"
MAX_IMAGES = 50  # Set to None for full dataset, or a small number for testing

# Optimal thresholds determined through threshold optimization study (Nov 27, 2025)
# BOX_THRESHOLD=0.1, TEXT_THRESHOLD=0.2 improved OWLv2 performance:
#   - AP@0.50: 0.277 → 0.287 (+3.6%)
#   - AR: 0.217 → 0.325 (+50%)
# Lower threshold captures valid low-confidence detections (0.1-0.25 range)
BOX_THRESHOLD = 0.1
TEXT_THRESHOLD = 0.2
IMG_SIZE = 640  # For YOLO-World resizing (optional for others)

# Paths
# Assuming running from notebooks/ or scripts/ and data is in ../
IMAGES_DIR = "../images_download"
ANNOTATIONS_DIR = "../annotated_images_download"

# Target Classes (from user request)
TARGET_CLASSES = [
    "Sump", "Cement Tank", "Plastic Barrel", "Metal Drum", "Mud Pot",
    "Plastic Bucket", "Stone Cistern", "Grinding-stone", "Cement Tanks",
    "Water Puddle", "Plant-holder", "Tyre", "Solid Waste", "Other Container"
]

# Prompt Formatting
PROMPT_LIST = TARGET_CLASSES
PROMPT_STRING = " . ".join(TARGET_CLASSES) + " ."

# --- Data Loading & COCO Conversion ---

def load_dataset_coco(images_dir, annotations_dir, max_images=None):
    """Reads Label Studio JSONs and creates an in-memory COCO dataset."""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {}
    }

    # Create categories map
    cat_name_to_id = {name: i+1 for i, name in enumerate(TARGET_CLASSES)}
    for name, id in cat_name_to_id.items():
        coco_data["categories"].append({"id": id, "name": name})

    json_files = glob.glob(os.path.join(annotations_dir, "*")) # Assuming files have no ext or .json
    if max_images:
        json_files = json_files[:max_images]

    ann_id_counter = 1
    valid_images = []

    print(f"Loading {len(json_files)} annotation files...")

    # Debug: Check categories and prompts
    print("COCO categories:", cat_name_to_id)
    print("PROMPT_LIST:", PROMPT_LIST)

    for jf in tqdm(json_files):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)

            # Extract image info
            task = data.get("task", {})
            img_url = task.get("data", {}).get("image", "")
            img_name = os.path.basename(img_url)
            img_path = os.path.join(images_dir, img_name)

            if not os.path.exists(img_path):
                continue

            # Get dimensions (Label Studio usually provides original_width/height in result)
            width, height = 0, 0
            results = data.get("result", [])
            if results:
                width = results[0].get("original_width", 0)
                height = results[0].get("original_height", 0)

            # If dimensions missing, read image (slow but necessary)
            if width == 0 or height == 0:
                im = Image.open(img_path)
                width, height = im.size

            image_id = task.get("id", 0)
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": width,
                "height": height,
                "absolute_path": img_path
            })

            # Process annotations
            for res in results:
                if res.get("type") == "rectanglelabels":
                    val = res.get("value", {})
                    labels = val.get("rectanglelabels", [])
                    x = val.get("x", 0) * width / 100.0
                    y = val.get("y", 0) * height / 100.0
                    w = val.get("width", 0) * width / 100.0
                    h = val.get("height", 0) * height / 100.0

                    for label in labels:
                        # Map dataset label to target class if possible, else skip or map to 'Other'
                        # Simple direct mapping for now based on user request
                        cat_id = cat_name_to_id.get(label)
                        if cat_id:
                            coco_data["annotations"].append({
                                "id": ann_id_counter,
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                                "iscrowd": 0
                            })
                            ann_id_counter += 1

        except Exception as e:
            print(f"Error processing {jf}: {e}")
            continue

    return coco_data, cat_name_to_id

# --- Model Wrappers ---

def init_detector(backend, device):
    print(f"Initializing {backend}...")
    if backend == "gdino_swint":
        return load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth", device=device)
    elif backend == "gdino_swinb":
        return load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "groundingdino_swinb_cogcoor.pth", device=device)
    elif backend == "owlv2":
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
        return (processor, model)
    elif backend == "yolo_world":
        model = YOLOWorld('yolov8s-world.pt')
        model.set_classes(PROMPT_LIST)
        return model
    else:
        raise ValueError(f"Unknown backend: {backend}")

def run_detector(backend, model, image_path, prompt_text, prompt_list, threshold, device):
    img = cv2.imread(image_path)
    if img is None: return [], [], []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if backend.startswith("gdino"):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_pil = Image.fromarray(img_rgb)
        img_tensor, _ = transform(img_pil, None)

        boxes, logits, phrases = groundingdino_predict(
            model=model,
            image=img_tensor,
            caption=prompt_text,
            box_threshold=threshold,
            text_threshold=TEXT_THRESHOLD,
            device=device
        )
        # Convert cxcywh to xywh and un-normalize
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes[:, :2] -= boxes[:, 2:] / 2  # cx,cy -> x,y
        return boxes.cpu().numpy(), logits.cpu().numpy(), phrases

    elif backend == "owlv2":
        processor, owl_model = model
        # Format prompts for OWLv2
        texts = [[f"a photo of a {t}" for t in prompt_list]]
        inputs = processor(text=texts, images=Image.fromarray(img_rgb), return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

        # OWLv2 returns xyxy, convert to xywh
        boxes = results["boxes"].cpu().numpy()
        boxes[:, 2] -= boxes[:, 0] # w = x2 - x1
        boxes[:, 3] -= boxes[:, 1] # h = y2 - y1

        scores = results["scores"].cpu().numpy()
        labels = [prompt_list[i] for i in results["labels"].cpu().numpy()]
        return boxes, scores, labels

    elif backend == "yolo_world":
        results = model.predict(img_rgb, conf=threshold, imgsz=IMG_SIZE, device=device, verbose=False)
        res = results[0]

        # YOLO returns xyxy, convert to xywh
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        labels = [prompt_list[i] for i in cls_ids]
        return boxes, scores, labels

    return [], [], []

# --- Main Evaluation Loop ---

def main():
    # 1. Load Data
    print("Loading dataset...")
    coco_dict, cat_map = load_dataset_coco(IMAGES_DIR, ANNOTATIONS_DIR, MAX_IMAGES)
    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()

    # 2. Initialize Model
    model = init_detector(MODEL_BACKEND, device)

    # 3. Run Inference
    coco_results = []
    inference_times = []

    print(f"Running evaluation for {MODEL_BACKEND} on {len(coco_dict['images'])} images...")

    for img_info in tqdm(coco_dict["images"]):
        start_time = time.time()

        boxes, scores, labels = run_detector(
            MODEL_BACKEND,
            model,
            img_info["absolute_path"],
            PROMPT_STRING,
            PROMPT_LIST,
            BOX_THRESHOLD,
            device
        )

        inference_times.append(time.time() - start_time)

        for box, score, label in zip(boxes, scores, labels):
            # Map label string back to category ID
            # Note: GDINO returns phrases, we need to match them to our classes
            # Simple substring match for now if exact match fails
            cat_id = cat_map.get(label)
            if not cat_id:
                # Try fuzzy match for GDINO output
                for name, cid in cat_map.items():
                    if name.lower() in label.lower() or label.lower() in name.lower():
                        cat_id = cid
                        break

            if cat_id:
                coco_results.append({
                    "image_id": img_info["id"],
                    "category_id": cat_id,
                    "bbox": box.tolist(),
                    "score": float(score)
                })

    # 4. Compute Metrics
    if not coco_results:
        print("No detections made!")
    else:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Runtime Stats
        avg_time = np.mean(inference_times)
        print(f"\n--- Runtime Stats ({MODEL_BACKEND}) ---")
        print(f"Average Inference Time: {avg_time:.4f} s/image")
        print(f"Throughput: {1/avg_time:.2f} FPS")

if __name__ == "__main__":
    main()
