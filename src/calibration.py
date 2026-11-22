import numpy as np, pandas as pd, json
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LOGITS_DIR = "results/calibration/val_logits"
PLANTDOC_DIR = "experiments/plantdoc-field-test"
OUT = "results/calibration"



y = np.load(f"{LOGITS_DIR}/val_labels.npy")
idx = np.arange(len(y))
fit_idx, eval_idx = train_test_split(idx, test_size=0.5, stratify=y, random_state=42)
pd.DataFrame({"index": np.r_[fit_idx, eval_idx], "half": ["fit"]*len(fit_idx)+["eval"]*len(eval_idx)}).to_csv(f"{OUT}/val_half_split.csv", index=False)

def ece(p, yy, bins=10):
    conf, pred = p.max(1), p.argmax(1); acc = pred == yy
    edges = np.linspace(0, 1, bins+1); e = 0; rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean()*abs(acc[m].mean()-conf[m].mean()); rows.append((lo, hi, m.sum(), conf[m].mean(), acc[m].mean()))
    return e, rows
def nll(T, lg, yy): return -log_softmax(lg/T, axis=1)[np.arange(len(yy)), yy].mean()
def fitT(lg, yy): return minimize_scalar(lambda t: nll(t, lg, yy), bounds=(0.05, 10), method="bounded").x

res, Ts, bins_store = [], {}, {}
for m in ("resnet50", "mobilenetv3"):
    for s in (42, 123, 2026):
        lg = np.load(f"{LOGITS_DIR}/{m}_seed{s}_val_logits.npy"); T = fitT(lg[fit_idx], y[fit_idx])
        e0, b0 = ece(softmax(lg[eval_idx], 1), y[eval_idx]); e1, b1 = ece(softmax(lg[eval_idx]/T, 1), y[eval_idx])
        n0, n1 = nll(1, lg[eval_idx], y[eval_idx]), nll(T, lg[eval_idx], y[eval_idx])
        acc = (lg[eval_idx].argmax(1) == y[eval_idx]).mean()
        res.append(dict(model=m, seed=s, T=T, accuracy=acc, ECE_before=e0, ECE_after=e1, NLL_before=n0, NLL_after=n1,
                        mean_conf_before=softmax(lg[eval_idx],1).max(1).mean()))
        Ts[f"{m}_seed{s}"] = float(T); bins_store[(m, s)] = (b0, b1)
r = pd.DataFrame(res); r.round(5).to_csv(f"{OUT}/calibration_per_seed.csv", index=False)
summ = r.groupby("model")[["T","accuracy","ECE_before","ECE_after","NLL_before","NLL_after"]].agg(["mean","std"]).round(5)
summ.to_csv(f"{OUT}/calibration_summary.csv"); json.dump(Ts, open(f"{OUT}/temperature_values.json","w"), indent=1)

fig, ax = plt.subplots(2, 2, figsize=(9, 8))
for i, m in enumerate(("resnet50", "mobilenetv3")):
    for j, (lab) in enumerate(("before", "after")):
        a = ax[i, j]; a.plot([0,1],[0,1],"k--",lw=1)
        for s, c in zip((42,123,2026), ("C0","C1","C2")):
            b = bins_store[(m, s)][j]; a.plot([x[3] for x in b],[x[4] for x in b],"o-",color=c,ms=4,label=f"seed {s}")
        a.set_title(f"{m} ({lab} temperature scaling)"); a.set_xlabel("mean confidence"); a.set_ylabel("accuracy"); a.legend(); a.set_xlim(0.3,1.01); a.set_ylim(0.3,1.01)
plt.tight_layout(); plt.savefig(f"{OUT}/reliability_diagrams.png", dpi=150)
print(r.round(4).to_string()); print(summ.to_string())
