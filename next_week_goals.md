# Next goals: TerraVista (final week)

Decide among yourselves who does what. The test, PlantDoc and data-efficiency results are good, so this week is about consistent numbers, the missing parts and a clean handover. Every number in the repo must come from a run or script whose output you can show.

1. **One set of numbers.** Both models (ResNet-50 and MobileNetV3) use the same final recipe, `uniform_p010`, in one config file. Make one table for both: validation macro-F1, test macro-F1 (name the official run), PlantDoc field macro-F1, parameters, file size and time per image (see below). Delete the hand-typed comparison table and any README number no file produces. Keep one test result per model: delete the files of the second MobileNetV3 test evaluation, and say in the README which run is the official one.
2. **Redo the error analysis from saved predictions.** For both models, on validation: weakest classes, most confused pairs, how many images both get wrong and how many only one gets wrong, and a few example images of the top confusions.
3. **Calibration.** Reliability diagram and ECE for both models, and temperature scaling fitted on half of validation and scored on the other half, before and after (see below). Also show the confidence on PlantDoc images.
4. **Grad-CAM.** Heatmaps for 30 to 50 fixed images (correct, wrong from the weakest classes, and some PlantDoc images). Say for each group whether the model looks at the lesion or the background.
5. **Streamlit app.** Upload a leaf photo, choose a model, and see the class, the calibrated confidence and the Grad-CAM heatmap, with "unsure" below a confidence threshold chosen on validation. A second page shows the results tables. It must start from the README instructions. Record a short screen video as a backup.
6. **Presentation and deployment brief.** About 12 slides, and a one-page brief for an agricultural extension service: which model, its size and speed, the "unsure" rule, what happens for crops outside the 38 classes, and the limitations. Say plainly whether the system is ready for the field.
7. **Clean repository.** Keep `README.md`, `src/`, `configs/`, `results/`, `app/`, `docs/` and one `notebooks/` folder. Delete `weekly report/`, the Week 3 test files, placeholder files, result zips, duplicate folders and unused large notebooks. The README has the data, setup, how to run each step, the results table and the limitations. Every file it names exists, and `verify_handoff.py` passes from a clean clone.

## How to measure size and speed (goal 1)

- **Parameters:** count them in the loaded 38-class model, not the standard ImageNet one.
- **File size:** the saved checkpoint.
- **Time per image:** median of 100 single-image runs after a warm-up, on CPU with one thread and on GPU with batch 64, on the same machine for both models.

## Calibration (goal 3)

- Put predictions into 10 confidence bins. In each bin compare the average confidence with how often the model is right. The reliability diagram plots this, and ECE is the average gap.
- Temperature scaling divides the model's outputs by one number T so confidence matches accuracy. It doesn't change predictions.
- This needs the saved checkpoints. Keep them somewhere shared (a Kaggle dataset, Drive or the Hugging Face Hub) and put the link in the README. Don't commit large model files to GitHub (100 MB limit).

## How to build and deploy the app (goal 5)

Streamlit is a Python library that turns a script into a web page, which updates when the user clicks.

**Build**
- Get the prediction working as a plain script first, then wrap it in a Streamlit page.
- Load each model once at start, use the same preprocessing as in training, and apply the temperature from goal 3.

**Deploy (free Streamlit Community Cloud)**
1. **Prepare the repo.** The app file is in `app/`, and `requirements.txt` at the top lists every library the app needs, with versions. Keep the list short, and use the CPU-only version of PyTorch, because the free server has no GPU and limited memory.
2. **Host the models outside GitHub.** ResNet-50 is about 90 MB, which is too close to GitHub's 100 MB limit. Upload both models to the Hugging Face Hub (free) or a shared Drive link, and make the app download each one the first time it starts and reuse the saved copy after that.
3. **Test from scratch on your own machine.** Make a new empty Python environment, install only from `requirements.txt`, and run the app. This catches missing libraries before the server does.
4. **Push to GitHub.** The repo must be public (or the server must have access), and the app must be on the branch you will deploy. Never commit passwords or tokens.
5. **Deploy.** Go to share.streamlit.io, sign in with GitHub, choose "Create app", pick the repo, the branch and the app file, and press Deploy. The first build takes a few minutes. If it fails, read the build log. The usual causes are a library missing from `requirements.txt` or running out of memory.
6. **Keep memory low.** Load only the model the user selected, not both, and make MobileNetV3 the default because it is smaller. Resize uploaded photos before processing.
7. **Before the demo.** Free apps go to sleep when nobody uses them, so open the link ten minutes before and run one prediction to wake it. Keep the laptop version and the screen video ready in case the link fails.
8. **Write it in the README:** the public link, the command to run it locally, and where the models are downloaded from.
9. **If Community Cloud keeps failing,** Hugging Face Spaces also hosts Streamlit apps for free, using the same files. A laptop demo with the video backup is still acceptable.
