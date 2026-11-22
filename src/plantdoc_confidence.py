import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LOGITS_DIR = "results/calibration/val_logits"
PLANTDOC_DIR = "experiments/plantdoc-field-test"
OUT = "results/calibration"

def ece(conf, acc, bins=10):
    edges = np.linspace(0,1,bins+1); e=0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.sum(): e+=m.mean()*abs(acc[m].mean()-conf[m].mean())
    return e
rows=[]; D={}
for m,tag in (("resnet50","resnet50"),("mobilenetv3","mobilenetv3_large")):
    for s in (42,123,2026):
        d=pd.read_csv(f"{PLANTDOC_DIR}/plantdoc_{tag}_seed{s}_predictions.csv"); acc=(d.y_true==d.y_pred).values; c=d.softmax_score.values
        D[(m,s)]=(c,acc)
        wrong=c[~acc]; right=c[acc]
        rows.append(dict(model=m,seed=s,n=len(d),accuracy=acc.mean(),mean_confidence=c.mean(),
            mean_conf_correct=right.mean(),mean_conf_wrong=wrong.mean(),ECE=ece(c,acc),
            frac_wrong_with_conf_ge_0_9=(wrong>=0.9).sum()/max(len(wrong),1)))
r=pd.DataFrame(rows); r.round(4).to_csv(f"{OUT}/plantdoc_calibration_per_seed.csv",index=False)
s=r.groupby("model").mean(numeric_only=True).drop(columns="seed").round(4); s.to_csv(f"{OUT}/plantdoc_calibration_summary.csv")
print(r.round(3).to_string()); print(s.to_string())
# threshold sweep: coverage vs accuracy (pooled over seeds)
sw=[]
for m in ("resnet50","mobilenetv3"):
    for t in (0.5,0.6,0.7,0.8,0.9,0.95):
        cov=[];ac=[]
        for sd in (42,123,2026):
            c,a=D[(m,sd)]; k=c>=t; cov.append(k.mean()); ac.append(a[k].mean() if k.sum() else np.nan)
        sw.append(dict(model=m,threshold=t,coverage=np.mean(cov),accuracy_on_kept=np.nanmean(ac)))
pd.DataFrame(sw).round(4).to_csv(f"{OUT}/plantdoc_threshold_sweep.csv",index=False); print(pd.DataFrame(sw).round(3).to_string())
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
for i,m in enumerate(("resnet50","mobilenetv3")):
    c=np.concatenate([D[(m,sd)][0] for sd in (42,123,2026)]); a=np.concatenate([D[(m,sd)][1] for sd in (42,123,2026)])
    ax[i].hist([c[a],c[~a]],bins=np.linspace(0,1,11),stacked=True,label=["correct","wrong"],color=["tab:green","tab:red"])
    ax[i].set_title(f"{m}: PlantDoc confidence (3 seeds pooled)"); ax[i].set_xlabel("confidence"); ax[i].set_ylabel("images"); ax[i].legend()
    edges=np.linspace(0,1,11); xs=[];ys=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        k=(c>lo)&(c<=hi)
        if k.sum()>=10: xs.append(c[k].mean()); ys.append(a[k].mean())
    ax[2].plot(xs,ys,"o-",label=m)
ax[2].plot([0,1],[0,1],"k--"); ax[2].set_title("PlantDoc reliability (bins with >=10 images)"); ax[2].set_xlabel("mean confidence"); ax[2].set_ylabel("accuracy"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/plantdoc_confidence.png",dpi=150)
