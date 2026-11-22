"""Run the 6 final checkpoints on the validation split and save raw logits (run on Kaggle with a GPU).
Inputs : PlantVillage dataset (color folder), the checkpoints dataset with train_val_split.csv and class_to_idx.json.
Outputs: val_labels.npy and <model>_seed<seed>_val_logits.npy (7,976 x 38) in OUT.
Only validation images are read; the test set is never touched."""
import json, glob, os, numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torchvision import models
from sklearn.metrics import f1_score

PV = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset"
UP = "/kaggle/input/datasets/shahnawaz006/terravista-ckpts"
OUT = "/kaggle/working/val_logits"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

hits = [p for p in glob.glob(f"{PV}/**/Apple___Apple_scab", recursive=True) if "color" in p.lower()]
assert hits, "Could not find color/Apple___Apple_scab under PlantVillage input"
COLOR_DIR = os.path.dirname(hits[0]); print("COLOR_DIR =", COLOR_DIR)

c2i = json.load(open(glob.glob(f"{UP}/**/class_to_idx.json", recursive=True)[0]))
df = pd.read_csv(glob.glob(f"{UP}/**/train_val_split.csv", recursive=True)[0])
df = df[df.split == "val"].reset_index(drop=True)
y = df["class"].map(c2i).values
assert len(df) == 7976 and not df["class"].map(c2i).isna().any()
np.save(f"{OUT}/val_labels.npy", y)

mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
class DS(torch.utils.data.Dataset):
    def __len__(self): return len(df)
    def __getitem__(self, i):
        im = Image.open(f"{COLOR_DIR}/{df.path[i]}").convert("RGB").resize((224,224), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
dl = torch.utils.data.DataLoader(DS(), batch_size=64, num_workers=2)

def build(name):
    if name == "resnet50":
        m = models.resnet50(); m.fc = nn.Linear(2048, 38)
    else:
        m = models.mobilenet_v3_large(); m.classifier[3] = nn.Linear(1280, 38)
    return m

def load(m, path):
    sd = torch.load(path, map_location="cpu")
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict): sd = sd[k]; break
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True); return m

runs = {}
for s in (42, 123, 2026):
    runs[("resnet50", s)] = glob.glob(f"{UP}/**/frac100_seed{s}/best_model.pt", recursive=True)[0]
    runs[("mobilenetv3", s)] = glob.glob(f"{UP}/**/*uniform_p010_seed{s}/best_weights.pt", recursive=True)[0]

for (name, s), p in runs.items():
    m = load(build(name), p).to(dev).eval(); out = []
    with torch.no_grad():
        for xb in dl:
            xb = ((xb.float()/255 - mean) / std).to(dev)
            out.append(m(xb).float().cpu())
    lg = torch.cat(out).numpy(); np.save(f"{OUT}/{name}_seed{s}_val_logits.npy", lg)
    print(name, s, "val macro-F1 =", round(f1_score(y, lg.argmax(1), average="macro"), 4))
