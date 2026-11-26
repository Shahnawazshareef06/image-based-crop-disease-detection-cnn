# image-based-crop-disease-detection-cnn
IIITH Minor Degree Program – Project 8: Image-Based Crop Disease Detection Using Convolutional Networks. A collaborative team repository for sharing code, documenting weekly progress, and tracking project contributions.

## Run the leaf classifier

Use Python 3.12. From the folder containing this README, create a fresh environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Open the local URL printed by Streamlit. **Results** opens immediately. For **Leaf analysis**, configure access to the model files below and upload a JPEG, PNG or WebP.

### Model access

The original weights are in the team's [TerraVista checkpoints on Kaggle](https://kaggle.com/datasets/3c624b2b02c92dff7c68454d82fe0d51e0685fb12dc2fafeece8f1257eb13bfd). The app requests one file from dataset version 1 and verifies it against the project's SHA-256 manifest before loading it. The Kaggle API currently requires an account with access to this dataset.

- **Kaggle download:** Set `KAGGLE_API_TOKEN` in your environment. Legacy `KAGGLE_USERNAME` and `KAGGLE_KEY` are also supported. On Community Cloud, enter credentials as top-level keys under app **Secrets**. Keep them outside Git.
- **Direct download:** If the team mirrors the exact files to Hugging Face, set `MOBILENETV3_LARGE_URL` and `RESNET50_URL` to the corresponding direct file URLs.
- **Offline laptop:** Download and extract the Kaggle ZIP once. Copy the two files below into `~/.cache/terravista/` (Windows: `%USERPROFILE%\.cache\terravista\`) under the specified cache names. Set `LEAF_MODEL_CACHE` to use a different directory.

| Model | Source path ending inside the dataset | Local cache filename |
|---|---|---|
| MobileNetV3-Large | `mobilenetv3_large_uniform_p010_seed42/best_weights.pt` | `mobilenetv3_large_seed42.pt` |
| ResNet-50 | `frac100_seed42/best_model.pt` | `resnet50_seed42.pt` |

### Predict from the command line

```bash
python app/predict.py leaf.jpg --model mobilenetv3_large --heatmap overlay.png
```

Substitute the path to your leaf photo. Use `--model resnet50` for the other checkpoint. The script prints the top class, decision, calibrated confidence, temperature and threshold as JSON.

Both entry points use RGB, PIL bilinear resizing to 224 × 224 and ImageNet normalization from validation. Probabilities use softmax(logits / T) with the selected seed-42 temperature. A score below 0.80 displays **unsure**, following the [calibration handover](docs/Calibration/README.md). Grad-CAM highlights contributions to the top class from the last spatial features. The results page reads the published lab and field CSVs directly.

### Deploy on Streamlit Community Cloud

Public app URL: pending deployment.

1. At [Streamlit Community Cloud](https://share.streamlit.io/), create an app from `dev-astitva/image-based-crop-disease-detection-cnn`, branch `streamlit-leaf-app`, entry point `app/streamlit_app.py`. Select Python 3.12 in Advanced settings and configure model access under Secrets. After merge, the upstream repository and merged branch can also be used.
2. Deploy, open the public URL, upload a leaf and check both model choices. Add the verified public URL above. The CPU build keeps one selected model cached in memory and downloaded checkpoints on disk for the lifetime of the instance.
3. Before presenting, open the app ten minutes early and run a prediction. Keep a local copy with the weights already cached and a recording of a successful prediction for backup.
