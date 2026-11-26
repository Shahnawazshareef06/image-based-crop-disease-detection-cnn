"""CPU inference and Grad-CAM for the two final TerraVista seed-42 checkpoints."""
import argparse
import base64
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"mobilenetv3_large": "MobileNetV3-Large", "resnet50": "ResNet-50"}
THRESHOLD = 0.80
DATASET = "shahnawaz006/terravista-ckpts"
API = "https://api.kaggle.com/v1/datasets"
PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def model_info(name):
    manifest = ROOT / "experiments/plantdoc-field-test/provenance.json"
    runs = json.loads(manifest.read_text())["runs"]
    return next(r for r in runs if r["model"] == name and r["seed"] == 42)


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def kaggle_request(url):
    request = Request(url)
    if os.getenv("KAGGLE_API_TOKEN"):
        request.add_unredirected_header("Authorization", "Bearer " + os.environ["KAGGLE_API_TOKEN"])
    elif os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        pair = os.environ["KAGGLE_USERNAME"] + ":" + os.environ["KAGGLE_KEY"]
        request.add_unredirected_header("Authorization", "Basic " + base64.b64encode(pair.encode()).decode())
    return urlopen(request, timeout=90)


def remote_path(name, suffix):
    token = ""
    while True:
        query = {"datasetVersionNumber": 1, "pageSize": 100}
        if token:
            query["pageToken"] = token
        with kaggle_request(f"{API}/list/{DATASET}?" + urlencode(query)) as response:
            page = json.load(response)
        for entry in page.get("datasetFiles", page.get("files", [])):
            path = entry.get("name", "")
            if path.endswith(suffix):
                return path
        token = page.get("nextPageToken")
        if not token:
            raise ValueError(f"Seed-42 weights for {MODEL_NAMES[name]} missing from dataset v1.")


def checkpoint(name):
    info = model_info(name)
    digest = info["checkpoint_sha256"]
    cache = Path(os.getenv("LEAF_MODEL_CACHE", Path.home() / ".cache/terravista"))
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{name}_seed42.pt"
    if dest.exists() and sha256(dest) == digest:
        return dest
    override = os.getenv(name.upper() + "_URL")
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        raw = Path(tmp) / "download"
        if override:
            source = urlopen(override, timeout=90)
        else:
            suffix = "/".join(info["checkpoint_path"].split("/")[-2:])
            path = remote_path(name, suffix)
            source = kaggle_request(f"{API}/download/{DATASET}/{quote(path, safe='')}?datasetVersionNumber=1")
        with source, raw.open("wb") as out:
            shutil.copyfileobj(source, out)
        if sha256(raw) != digest and zipfile.is_zipfile(raw):
            with zipfile.ZipFile(raw) as archive:
                members = [m for m in archive.infolist()
                           if Path(m.filename).name == Path(info["checkpoint_path"]).name]
                if len(members) == 1:
                    unpacked = Path(tmp) / "unpacked"
                    with archive.open(members[0]) as src, unpacked.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    raw = unpacked
        if sha256(raw) != digest:
            raise ValueError("Checkpoint checksum mismatch. Check the seed and download URL.")
        raw.replace(dest)
    return dest


def load_model(name):
    path = checkpoint(name)
    classes = json.loads((ROOT / "models/mobilenetv3_large/class_to_idx.json").read_text())
    labels = sorted(classes, key=classes.get)
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    else:
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(labels))
    weights = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(weights.get(key), dict):
            weights = weights[key]
            break
    model.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()})
    temperatures = json.loads((ROOT / "Results/Calibration/temperature_values.json").read_text())
    key = name.removesuffix("_large") + "_seed42"
    return model.eval().requires_grad_(False), labels, temperatures[key]


def prepare(source):
    with Image.open(source) as im:
        if im.width * im.height > 24_000_000:
            raise ValueError("Choose a photo with at most 24 megapixels.")
        return ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)


def predict(loaded, image):
    model, labels, temperature = loaded
    x = PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        if isinstance(model, models.ResNet):
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            features = model.layer4(model.layer3(model.layer2(model.layer1(x))))
            head = model.fc
        else:
            features = model.features(x)
            head = model.classifier
    with torch.enable_grad():
        features = features.detach().requires_grad_(True)
        logits = head(model.avgpool(features).flatten(1))[0]
        probs = (logits.detach() / temperature).softmax(0)
        idx = int(probs.argmax())
        gradient, = torch.autograd.grad(logits[idx], features)
    cam = (gradient.mean((2, 3), keepdim=True) * features.detach()).sum(1).clamp_min(0)
    cam = cam - cam.min()
    peak = float(cam.max())
    if peak:
        cam /= peak
    mask = transforms.ToPILImage()(cam).resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.composite(Image.new("RGB", image.size, "#ff4900"), image,
                              mask.point(lambda v: round(v * 0.65)))
    confidence = float(probs[idx])
    return {"class": labels[idx], "decision": labels[idx] if confidence >= THRESHOLD else "unsure",
            "confidence": confidence, "temperature": temperature, "threshold": THRESHOLD}, overlay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, default="mobilenetv3_large")
    parser.add_argument("--heatmap", type=Path)
    args = parser.parse_args()
    try:
        photo = prepare(args.image)
        result, overlay = predict(load_model(args.model), photo)
        if args.heatmap:
            overlay.save(args.heatmap)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError, URLError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
