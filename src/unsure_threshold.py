import numpy as np, pandas as pd, json
from scipy.special import softmax
LOGITS_DIR = "results/calibration/val_logits"
PLANTDOC_DIR = "experiments/plantdoc-field-test"
OUT = "results/calibration"

y=np.load(f"{LOGITS_DIR}/val_labels.npy"); sp=pd.read_csv(f"{OUT}/val_half_split.csv")
fit=sp[sp.half=="fit"]["index"].values; ev=sp[sp.half=="eval"]["index"].values
Ts=json.load(open(f"{OUT}/temperature_values.json"))
def pick(conf,acc,target):
    # lowest threshold t such that accuracy on kept (conf>=t) >= target; scan candidate confidences
    order=np.argsort(conf); c=conf[order]; a=acc[order]
    # accuracy of kept set for each cutoff i = a[i:].mean()
    tail=np.cumsum(a[::-1])[::-1]/np.arange(len(a),0,-1)
    ok=np.where(tail>=target)[0]
    return c[ok[0]] if len(ok) else 1.0
rows=[]
for m in ("resnet50","mobilenetv3"):
    for s in (42,123,2026):
        lg=np.load(f"{LOGITS_DIR}/{m}_seed{s}_val_logits.npy"); p=softmax(lg/Ts[f"{m}_seed{s}"],1)
        conf,acc=p.max(1),p.argmax(1)==y
        for tgt in (0.99,0.995):
            t=pick(conf[fit],acc[fit],tgt); k=conf[ev]>=t
            rows.append(dict(model=m,seed=s,target_accuracy=tgt,threshold=round(float(t),4),
                coverage_eval=round(k.mean(),4),accuracy_on_kept_eval=round(acc[ev][k].mean(),4),
                unsure_wrong_caught=round((~acc[ev] & ~k).sum()/max((~acc[ev]).sum(),1),4)))
r=pd.DataFrame(rows); r.to_csv(f"{OUT}/unsure_threshold_per_seed.csv",index=False); print(r.to_string())
s=r.groupby(["model","target_accuracy"])[["threshold","coverage_eval","accuracy_on_kept_eval","unsure_wrong_caught"]].agg(["mean","min","max"]).round(4)
s.to_csv(f"{OUT}/unsure_threshold_summary.csv"); print(s.to_string())
# apply to PlantDoc (raw confidence, no T available)
out=[]
for m,tag in (("resnet50","resnet50"),("mobilenetv3","mobilenetv3_large")):
    for tgt in (0.99,0.995):
        t=r[(r.model==m)&(r.target_accuracy==tgt)].threshold.mean()
        ks=[];ac=[]
        for sd in (42,123,2026):
            d=pd.read_csv(f"{PLANTDOC_DIR}/plantdoc_{tag}_seed{sd}_predictions.csv"); k=d.softmax_score>=t
            ks.append(k.mean()); ac.append((d.y_true==d.y_pred)[k].mean() if k.sum() else np.nan)
        out.append(dict(model=m,target_accuracy=tgt,threshold=round(t,3),plantdoc_coverage=round(np.mean(ks),3),plantdoc_accuracy_on_kept=round(np.nanmean(ac),3)))
o=pd.DataFrame(out); o.to_csv(f"{OUT}/unsure_threshold_on_plantdoc.csv",index=False); print(o.to_string())
