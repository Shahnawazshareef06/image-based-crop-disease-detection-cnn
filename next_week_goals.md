# Next goals: TerraVista

Decide among yourselves who does what. The test results, the PlantDoc field test, the data-efficiency curve and the calibration are done. What remains is fixing a few results, building the app, preparing the presentation and the brief, and cleaning the repository. Every number in the repo must come from a run or script whose output you can show.

1. **One set of numbers.** Make one table in the README for both models (ResNet-50 and MobileNetV3, `uniform_p010`): validation macro-F1, test macro-F1, PlantDoc field macro-F1, parameters and file size (see below). Keep one test result per model: delete the files of the second MobileNetV3 test evaluation and say in the README which run is the official one. State which data and split file the models use, and make the files in `data/` match it (the deduplicated CSV in `data/duplicates/` covers only 27 classes, while the split covers 38), or label it as an earlier step.
2. **Fix the Grad-CAM analysis.** Use real PlantDoc test images for the "field" group (from the PlantDoc cohort file), give every image its own heatmap and result with no repeated values, and describe in the README how a heatmap is judged lesion-focused or background-focused. Keep one script.
3. **Error analysis across all three seeds.** The committed numbers match seed 42 only (the script analyses one seed per run, default 42). Run it for seeds 123 and 2026 too, and report the overlap between the models and the weakest classes across the three seeds.
4. **Streamlit app.** Upload a leaf photo, choose a model, and see the class, the calibrated confidence and the Grad-CAM heatmap, with "unsure" below the threshold from the calibration. A second page shows the results table. It must start from the README instructions. Record a short screen video as a backup (see below).
5. **Presentation.** About 15 slides, in this flow: the problem, the data and how it was cleaned, how results are measured, the two models and how they were trained, what augmentation and class imbalance showed, the data-efficiency curve, the test results with intervals, the lab-to-field gap on PlantDoc, calibration and the "unsure" rule, error analysis, Grad-CAM, the demo, deployment, and the limitations.
6. **Deployment brief.** A one-page brief for an agricultural extension service: model choice, size, the "unsure" rule, crops outside the 38 classes, and the limitations.
7. **Clean repository.** Keep `README.md`, `src/`, `models/`, `analysis/`, `experiments/`, `data/`, `docs/` and one `notebooks/` folder. Delete `weekly report/` (including the Week 3 test files), placeholder files, duplicate model folders and unused large notebooks. Use lowercase folder names throughout the repository. The README has the data, setup, how to run each step, the results table and the limitations. Every file it names exists, and `verify_handoff.py` passes from a clean clone.

## How to get parameters and file size (goal 1)

- **Parameters:** count them in the loaded 38-class model, not the standard ImageNet one.
- **File size:** the size of the saved checkpoint.

## How to build and deploy the app (goal 4)

Streamlit is a Python library that turns a script into a web page, which updates when the user clicks.

**Build**
- Get the prediction working as a plain script first, then wrap it in a Streamlit page.
- Load each model once at start, use the same preprocessing as in training, and apply the temperature from the calibration.

**Deploy (free Streamlit Community Cloud)**
1. **Prepare the repo.** The app file is in `app/`, and `requirements.txt` at the top lists every library the app needs, with versions. Keep the list short, and use the CPU-only version of PyTorch, because the free server has no GPU and limited memory.
2. **Host the models outside GitHub.** ResNet-50 is about 90 MB, which is too close to GitHub's 100 MB limit. Use the Kaggle dataset or the Hugging Face Hub (free), and make the app download each model the first time it starts and reuse the saved copy after that.
3. **Test from scratch on your own machine.** Make a new empty Python environment, install only from `requirements.txt`, and run the app. This catches missing libraries before the server does.
4. **Push to GitHub.** The repo must be public (or the server must have access), and the app must be on the branch you will deploy. Never commit passwords or tokens.
5. **Deploy.** Go to share.streamlit.io, sign in with GitHub, choose "Create app", pick the repo, the branch and the app file, and press Deploy. The first build takes a few minutes. If it fails, read the build log. The usual causes are a library missing from `requirements.txt` or running out of memory.
6. **Keep memory low.** Load only the model the user selected, not both, and make MobileNetV3 the default because it is smaller. Resize uploaded photos before processing.
7. **Before the demo.** Free apps go to sleep when nobody uses them, so open the link ten minutes before and run one prediction to wake it. Keep the laptop version and the screen video ready in case the link fails.
8. **Write it in the README:** the public link, the command to run it locally, and where the models are downloaded from.
9. **If Community Cloud keeps failing,** Hugging Face Spaces also hosts Streamlit apps for free, using the same files. A laptop demo with the video backup is still acceptable.
