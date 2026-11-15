# PlantDoc field evaluation

## Main comparison with duplicate screening

| Model | Screened lab macro-F1 | Screened field macro-F1 | Field minus lab (95% CI) |
|---|---:|---:|---:|
| ResNet-50 | 0.9806 ± 0.0020 [0.9770, 0.9840] | 0.2822 ± 0.0203 [0.2383, 0.3182] | -0.6983 [-0.7422, -0.6619] |
| MobileNetV3-Large | 0.9605 ± 0.0116 [0.9555, 0.9652] | 0.2320 ± 0.0056 [0.1866, 0.2676] | -0.7285 [-0.7741, -0.6927] |

Both sides cover the same 27 true classes. The main comparison uses 241 PlantDoc test images after excluding 11 flagged images and 5,169 PlantVillage test images after excluding 300 of 5,469 same-class images. The cross-dataset scan found zero exact matches and zero near-duplicate candidates under its rules.

The lab exclusions combine PlantVillage test-to-train flags from cosine similarity ≥ 0.95 or dHash distance ≤ 4 with Part 4's within-PlantVillage exact pixel matches and dual pHash/dHash candidates. Candidates are exclusions under a conservative policy. Similar images may be flagged, and substantial edits may evade the audit. Only the reporting cohort changes. All scores use saved predictions from the same six model runs, with three seeds per architecture. The official PlantVillage test results are preserved.

See `duplicate_screened_metrics.csv` for the table values, `duplicate_screened_images_per_class.csv` for support in both domains, and `lab_duplicate_exclusions.csv` for the flagged lab paths and evidence. The bootstrap and seed summary follow the method below. To reproduce this comparison, run the companion notebook in screened_report mode with the original split, the saved field outputs, the six official PlantVillage test prediction CSVs and test_near_duplicate_flags.csv.

## Original official-split lab reference

The lab numbers in this section precede duplicate screening. The field numbers already use the screened cohort.

| Model | Lab Macro-F1 | Field Macro-F1 | Field minus lab (95% CI) |
|---|---:|---:|---:|
| resnet50 | 0.9807 ± 0.0021 [0.9771, 0.9839] | 0.2822 ± 0.0203 [0.2383, 0.3182] | -0.6985 [-0.7431, -0.6623] |
| mobilenetv3_large | 0.9607 ± 0.0115 [0.9560, 0.9653] | 0.2320 ± 0.0056 [0.1866, 0.2676] | -0.7287 [-0.7743, -0.6931] |

Both domains are scored on the same 27 target classes. Field images: 241. Lab images: 5469. The lab cohort uses the official test prediction files filtered to the field-supported classes.

MobileNetV3-Large minus ResNet-50 on PlantDoc: -0.0502 [-0.0908, -0.0117] (paired image bootstrap).

Macro-F1 averages equally over true classes in the cohort. Predictions can use all 38 outputs. SD is the sample standard deviation over the three seeds. The 95% percentile CIs resample images within true class 4,000 times and share each resample across all six runs in a domain. The intervals condition on the fitted seeds. For lab vs. field, each domain is resampled independently.

## Overlap and exclusions

- PlantDoc test, `conflicting_plantdoc_duplicate`: 5 images.
- PlantDoc test, `possible_plantdoc_duplicate`: 5 images.
- PlantDoc test, `repeated_plantdoc_image`: 1 images.
- PlantDoc test, `retained`: 241 images.
- PlantDoc train, `conflicting_plantdoc_duplicate`: 5 images.
- PlantDoc train, `possible_plantdoc_duplicate`: 35 images.
- PlantDoc train, `repeated_plantdoc_image`: 2 images.
- PlantDoc train, `retained`: 2628 images.

| Check | Images scanned | Images with exact match | Images with near-duplicate candidate |
|---|---:|---:|---:|
| plantdoc_vs_plantvillage | 2922 | 0 | 0 |
| within_plantdoc | 2922 | 16 | 79 |
| within_plantvillage | 54305 | 42 | 76 |

Cross-dataset counts are the number of PlantDoc images with a PlantVillage match. Within-dataset counts include both members of a pair and may count an image in both columns. The audit checks exact decoded pixels and flags candidates when pHash and dHash distances are both at most four. Crops or substantial edits can escape screening. See `duplicate_summary.csv` for split-level counts and `duplicate_checks.csv` for per-image evidence. The field cohort excluded 11 of 252 test images under the policies listed above. The cross-dataset screen flagged zero images. The official PlantVillage split and saved lab predictions stay fixed.

## Selected field failure patterns

**resnet50**

- Apple___Cedar_apple_rust: mean class F1 0.000 (10 images).
- Potato___Late_blight: mean class F1 0.000 (7 images).
- Strawberry___healthy: mean class F1 0.000 (8 images).
- Tomato___Tomato_Yellow_Leaf_Curl_Virus predicted as Tomato___Late_blight: 21 errors summed over three seeds.
- Tomato___Tomato_mosaic_virus predicted as Tomato___Late_blight: 19 errors summed over three seeds.
- Strawberry___healthy predicted as Raspberry___healthy: 18 errors summed over three seeds.

**mobilenetv3_large**

- Cherry_(including_sour)___healthy: mean class F1 0.000 (10 images).
- Peach___healthy: mean class F1 0.000 (9 images).
- Potato___Late_blight: mean class F1 0.000 (7 images).
- Potato___Early_blight predicted as Tomato___Late_blight: 20 errors summed over three seeds.
- Tomato___Tomato_Yellow_Leaf_Curl_Virus predicted as Squash___Powdery_mildew: 17 errors summed over three seeds.
- Corn_(maize)___Common_rust_ predicted as Corn_(maize)___Northern_Leaf_Blight: 16 errors summed over three seeds.

The confusion counts describe observed errors. Inspect the matching image paths in the prediction CSVs before attributing errors to visual causes.
