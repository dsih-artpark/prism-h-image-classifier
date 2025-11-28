import os
import sys
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
from torchvision.ops import nms

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
MODEL_BACKEND = "ensemble_3"
MAX_IMAGES = 50

# Optimal thresholds determined through threshold optimization study (Nov 27, 2025)
# BOX_THRESHOLD=0.1, TEXT_THRESHOLD=0.2 improved performance:
#   OWLv2: AP@0.50 0.277→0.287 (+3.6%), AR 0.217→0.325 (+50%)
#   Ensemble: AP@0.50 0.140→0.142 (+1.4%), AR 0.233→0.252 (+8%)
BOX_THRESHOLD = 0.1
TEXT_THRESHOLD = 0.2
IMG_SIZE = 640

# Paths
IMAGES_DIR = "../images_download"
ANNOTATIONS_DIR = "../annotated_images_download"

# Target Classes
TARGET_CLASSES = [
    "Sump", "Cement Tank", "Plastic Barrel", "Metal Drum", "Mud Pot",
    "Plastic Bucket", "Stone Cistern", "Grinding-stone", "Cement Tanks",
    "Water Puddle", "Plant-holder", "Tyre", "Solid Waste", "Other Container"
]

# Prompt Formatting
PROMPT_LIST = TARGET_CLASSES
PROMPT_STRING = " . ".join(TARGET_CLASSES) + " ."

# --- Data Loading ---

def load_dataset_coco(images_dir, annotations_dir, max_images=None):
    """Reads Label Studio JSONs and creates an in-memory COCO dataset."""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {}
    }

    cat_name_to_id = {name: i+1 for i, name in enumerate(TARGET_CLASSES)}
    for name, id in cat_name_to_id.items():
        coco_data["categories"].append({"id": id, "name": name})

    json_files = glob.glob(os.path.join(annotations_dir, "*"))
    if max_images:
        json_files = json_files[:max_images]

    ann_id_counter = 1

    print(f"Loading {len(json_files)} annotation files...")
    for jf in tqdm(json_files):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)

            task = data.get("task", {})
            img_url = task.get("data", {}).get("image", "")
            img_name = os.path.basename(img_url)
            img_path = os.path.join(images_dir, img_name)

            if not os.path.exists(img_path):
                continue

            width, height = 0, 0
            results = data.get("result", [])
            if results:
                width = results[0].get("original_width", 0)
                height = results[0].get("original_height", 0)

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

            for res in results:
                if res.get("type") == "rectanglelabels":
                    val = res.get("value", {})
                    labels = val.get("rectanglelabels", [])
                    x = val.get("x", 0) * width / 100.0
                    y = val.get("y", 0) * height / 100.0
                    w = val.get("width", 0) * width / 100.0
                    h = val.get("height", 0) * height / 100.0

                    for label in labels:
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
    elif backend == "ensemble_3":
        # Load all three models
        models = {
            "gdino_swint": init_detector("gdino_swint", device),
            "gdino_swinb": init_detector("gdino_swinb", device),
            "owlv2":       init_detector("owlv2", device),
        }
        return models
    else:
        raise ValueError(f"Unknown backend: {backend}")

def run_single_detector(backend, model, image_path, prompt_text, prompt_list, threshold, device):
    """Helper to run a single detector and return standardized results."""
    img = cv2.imread(image_path)
    if img is None: return np.array([]), np.array([]), []
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
        if len(boxes) > 0:
            boxes = boxes * torch.Tensor([w, h, w, h])
            boxes[:, :2] -= boxes[:, 2:] / 2  # cx,cy -> x,y
            return boxes.cpu().numpy(), logits.cpu().numpy(), phrases
        else:
            return np.array([]), np.array([]), []

    elif backend == "owlv2":
        processor, owl_model = model
        texts = [[f"a photo of a {t}" for t in prompt_list]]
        inputs = processor(text=texts, images=Image.fromarray(img_rgb), return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

        boxes = results["boxes"].cpu().numpy()
        if len(boxes) > 0:
            boxes[:, 2] -= boxes[:, 0] # w = x2 - x1
            boxes[:, 3] -= boxes[:, 1] # h = y2 - y1

        scores = results["scores"].cpu().numpy()
        labels = [prompt_list[i] for i in results["labels"].cpu().numpy()]
        return boxes, scores, labels

    return np.array([]), np.array([]), []

def run_detector(backend, model, image_path, prompt_text, prompt_list, threshold, device):
    if backend == "ensemble_3":
        # Run all three models
        boxes_T, scores_T, labels_T = run_single_detector("gdino_swint", model["gdino_swint"], image_path, prompt_text, prompt_list, threshold, device)
        boxes_B, scores_B, labels_B = run_single_detector("gdino_swinb", model["gdino_swinb"], image_path, prompt_text, prompt_list, threshold, device)
        boxes_O, scores_O, labels_O = run_single_detector("owlv2",       model["owlv2"],       image_path, prompt_text, prompt_list, threshold, device)

        # Concatenate results
        all_boxes_list = []
        all_scores_list = []
        all_labels_list = []

        if len(boxes_T) > 0:
            all_boxes_list.append(boxes_T)
            all_scores_list.append(scores_T)
            all_labels_list.extend(labels_T)
        if len(boxes_B) > 0:
            all_boxes_list.append(boxes_B)
            all_scores_list.append(scores_B)
            all_labels_list.extend(labels_B)
        if len(boxes_O) > 0:
            all_boxes_list.append(boxes_O)
            all_scores_list.append(scores_O)
            all_labels_list.extend(labels_O)

        if not all_boxes_list:
            return np.array([]), np.array([]), []

        boxes_all = np.concatenate(all_boxes_list, axis=0)
        scores_all = np.concatenate(all_scores_list, axis=0)
        labels_all = all_labels_list

        # Apply NMS
        # Convert xywh to xyxy for NMS
        boxes_tensor = torch.tensor(boxes_all, dtype=torch.float32)
        boxes_xyxy = boxes_tensor.clone()
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]

        scores_tensor = torch.tensor(scores_all, dtype=torch.float32)

        # NMS threshold 0.5
        keep_idx = nms(boxes_xyxy, scores_tensor, iou_threshold=0.5)

        boxes_ens = boxes_tensor[keep_idx].numpy()
        scores_ens = scores_tensor[keep_idx].numpy()
        labels_ens = [labels_all[i] for i in keep_idx]

        return boxes_ens, scores_ens, labels_ens

    else:
        return run_single_detector(backend, model, image_path, prompt_text, prompt_list, threshold, device)

# --- Main Evaluation Loop ---

def main():
    print("Loading dataset...")
    coco_dict, cat_map = load_dataset_coco(IMAGES_DIR, ANNOTATIONS_DIR, MAX_IMAGES)
    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()

    model = init_detector(MODEL_BACKEND, device)

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
            cat_id = cat_map.get(label)
            if not cat_id:
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

    if not coco_results:
        print("No detections made!")
    else:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        avg_time = np.mean(inference_times)
        print(f"\n--- Runtime Stats ({MODEL_BACKEND}) ---")
        print(f"Average Inference Time: {avg_time:.4f} s/image")
        print(f"Throughput: {1/avg_time:.2f} FPS")

if __name__ == "__main__":
    main()
