# TerraVista PlantDoc class mapping

[plantdoc_to_terravista_mapping.csv](plantdoc_to_terravista_mapping.csv) maps the 28 class folders observed in the [PlantDoc Classification dataset on Kaggle](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset), version 7, to TerraVista’s 38-class PlantVillage label space. Use this mapping when building a PlantDoc evaluation set for the TerraVista classifier.

## Coverage

| PlantDoc split | Image files | Readable headers | Fully decoded audit samples | TerraVista classes covered |
|---|---:|---:|---:|---:|
| Train | 2,670 | 2,670 | 28 | 28 |
| Test | 252 | 252 | 27 | 27 |
| Combined | 2,922 | 2,922 | 55 | 28 |

- **Overall coverage is 28 of 38 classes (73.7%).** Ten TerraVista classes have no matching source folder in this release.
- **Test coverage is 27 of 38 classes (71.1%) before screening.**
- **Test support is uneven.** The represented classes have 4 to 15 test images each, with a median of 9. Corn gray leaf spot (ID 7) has the fewest test images, and tomato yellow leaf curl virus (ID 35) has the most. Report per-class support alongside Macro-F1 so readers can see which scores are based on few examples.
- **The counts come from the dataset audit.** All image headers were checked, and one image per populated class folder and split was fully decoded. The table records file counts before full evaluation decoding, duplicate removal, and screening for overlap with PlantVillage.

## Mapping decisions and class IDs

The CSV has **11 direct label matches** after name normalization and **17 project-reviewed mappings** supported by the published crosswalk. The evidence concerns class-label equivalence. Individual image diagnoses were outside the scope of the mapping audit.

- `source_label` is the PlantDoc folder name. `terravista_label` is the exact TerraVista label string, including punctuation, spaces, and underscores.
- `terravista_id` is the original zero-based model output index from TerraVista’s `class_to_idx.json`. Match by the canonical label and use the saved ID when evaluating. CSV row positions and PlantDoc folder ordering can differ from model indices.
- `decision` and `evidence` describe how the mapping was established. Split-prefixed columns record image file counts, readable headers, and fully decoded audit samples. A zero in an image file count means the class has no files in that split. The fully decoded count records the audit sample size.

## Unmapped TerraVista classes

These ten original output IDs have zero mapped PlantDoc files in both splits. They are listed in the coverage table and omitted from the source-folder mapping CSV.

| ID | TerraVista label |
|---:|---|
| 1 | `Apple___Black_rot` |
| 5 | `Cherry_(including_sour)___Powdery_mildew` |
| 10 | `Corn_(maize)___healthy` |
| 12 | `Grape___Esca_(Black_Measles)` |
| 13 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` |
| 15 | `Orange___Haunglongbing_(Citrus_greening)` |
| 16 | `Peach___Bacterial_spot` |
| 22 | `Potato___healthy` |
| 26 | `Strawberry___Leaf_scorch` |
| 34 | `Tomato___Target_Spot` |

**Mapped with no test images:** ID 33, `Tomato___Spider_mites Two-spotted_spider_mite`, has two train images. Together with the ten unmapped classes above, it accounts for the **11 TerraVista classes with zero test images** in this release.

## Full class coverage by TerraVista ID

Counts below are readable image headers from the audit. A dash marks a TerraVista label with no mapped PlantDoc folder.

| ID | TerraVista label | PlantDoc folder | Train | Test |
|---:|---|---|---:|---:|
| 0 | `Apple___Apple_scab` | `Apple_Scab_Leaf` | 83 | 10 |
| 1 | `Apple___Black_rot` | — | 0 | 0 |
| 2 | `Apple___Cedar_apple_rust` | `Apple_rust_leaf` | 96 | 10 |
| 3 | `Apple___healthy` | `Apple_leaf` | 79 | 9 |
| 4 | `Blueberry___healthy` | `Blueberry_leaf` | 106 | 11 |
| 5 | `Cherry_(including_sour)___Powdery_mildew` | — | 0 | 0 |
| 6 | `Cherry_(including_sour)___healthy` | `Cherry_leaf` | 47 | 10 |
| 7 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn_Gray_leaf_spot` | 63 | 4 |
| 8 | `Corn_(maize)___Common_rust_` | `Corn_rust_leaf` | 107 | 10 |
| 9 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_leaf_blight` | 182 | 12 |
| 10 | `Corn_(maize)___healthy` | — | 0 | 0 |
| 11 | `Grape___Black_rot` | `grape_leaf_black_rot` | 71 | 8 |
| 12 | `Grape___Esca_(Black_Measles)` | — | 0 | 0 |
| 13 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | — | 0 | 0 |
| 14 | `Grape___healthy` | `grape_leaf` | 63 | 12 |
| 15 | `Orange___Haunglongbing_(Citrus_greening)` | — | 0 | 0 |
| 16 | `Peach___Bacterial_spot` | — | 0 | 0 |
| 17 | `Peach___healthy` | `Peach_leaf` | 103 | 9 |
| 18 | `Pepper,_bell___Bacterial_spot` | `Bell_pepper_leaf_spot` | 74 | 9 |
| 19 | `Pepper,_bell___healthy` | `Bell_pepper_leaf` | 34 | 8 |
| 20 | `Potato___Early_blight` | `Potato_leaf_early_blight` | 157 | 14 |
| 21 | `Potato___Late_blight` | `Potato_leaf_late_blight` | 200 | 8 |
| 22 | `Potato___healthy` | — | 0 | 0 |
| 23 | `Raspberry___healthy` | `Raspberry_leaf` | 112 | 7 |
| 24 | `Soybean___healthy` | `Soyabean_leaf` | 57 | 8 |
| 25 | `Squash___Powdery_mildew` | `Squash_Powdery_mildew_leaf` | 124 | 6 |
| 26 | `Strawberry___Leaf_scorch` | — | 0 | 0 |
| 27 | `Strawberry___healthy` | `Strawberry_leaf` | 88 | 8 |
| 28 | `Tomato___Bacterial_spot` | `Tomato_leaf_bacterial_spot` | 101 | 9 |
| 29 | `Tomato___Early_blight` | `Tomato_Early_blight_leaf` | 79 | 9 |
| 30 | `Tomato___Late_blight` | `Tomato_leaf_late_blight` | 101 | 10 |
| 31 | `Tomato___Leaf_Mold` | `Tomato_mold_leaf` | 85 | 6 |
| 32 | `Tomato___Septoria_leaf_spot` | `Tomato_Septoria_leaf_spot` | 145 | 12 |
| 33 | `Tomato___Spider_mites Two-spotted_spider_mite` | `Tomato_two_spotted_spider_mites_leaf` | 2 | 0 |
| 34 | `Tomato___Target_Spot` | — | 0 | 0 |
| 35 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_leaf_yellow_virus` | 223 | 15 |
| 36 | `Tomato___Tomato_mosaic_virus` | `Tomato_leaf_mosaic_virus` | 44 | 10 |
| 37 | `Tomato___healthy` | `Tomato_leaf` | 44 | 8 |
