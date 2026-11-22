# Plant Leaf Disease Dataset: Train/Val/Test Split

`split.csv` assigns every image in a plant leaf disease image collection to a **train**, **val** or **test** split. The class names and file naming follow the PlantVillage convention (`Crop___Condition`). The CSV holds only the manifest, not the images themselves.

## Overview

| Property | Value |
|---|---|
| Images | 53,705 |
| Classes | 38 (crop + disease, or crop + healthy) |
| Splits | `train`, `val`, `test` |
| Duplicate paths | 0 |
| File types | 53,703 `.jpg`, 1 `.png`, 1 `.jpeg` |

## File format

`split.csv` is UTF-8 with a header row and three columns:

| Column | Type | Description |
|---|---|---|
| `path` | string | Image path relative to the dataset root, in the form `<class>/<filename>` |
| `class` | string | Class label. Identical to the first path component |
| `split` | string | One of `train`, `val`, `test` |

Notes on paths:

- Some filenames contain spaces, and some class names contain commas (`Pepper,_bell___healthy`). Use a real CSV parser, not `split(",")`.
- The class name uses `___` (three underscores) to separate crop from condition.
- Extensions are mixed case (`.JPG`, `.jpg`), so avoid case-sensitive globbing.

## Split sizes

| Split | Images | Share |
|---|---|---|
| train | 37,751 | 70.3% |
| val | 7,976 | 14.9% |
| test | 7,978 | 14.9% |

The split is roughly 70/15/15, and every one of the 38 classes appears in all three splits.

## Classes

The 38 classes cover 14 crops: apple, blueberry, cherry, corn (maize), grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry and tomato.

| Class | Images |
|---|---:|
| Orange___Haunglongbing_(Citrus_greening) | 5,487 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,329 |
| Soybean___healthy | 4,650 |
| Peach___Bacterial_spot | 2,296 |
| Tomato___Bacterial_spot | 2,126 |
| Tomato___Late_blight | 1,900 |
| Squash___Powdery_mildew | 1,834 |
| Tomato___Septoria_leaf_spot | 1,771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 |
| Apple___healthy | 1,638 |
| Tomato___healthy | 1,578 |
| Blueberry___healthy | 1,496 |
| Pepper,_bell___healthy | 1,468 |
| Tomato___Target_Spot | 1,404 |
| Grape___Esca_(Black_Measles) | 1,383 |
| Corn_(maize)___Common_rust_ | 1,192 |
| Grape___Black_rot | 1,180 |
| Corn_(maize)___healthy | 1,159 |
| Strawberry___Leaf_scorch | 1,107 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1,076 |
| Cherry_(including_sour)___Powdery_mildew | 1,039 |
| Tomato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___Early_blight | 999 |
| Pepper,_bell___Bacterial_spot | 995 |
| Corn_(maize)___Northern_Leaf_Blight | 983 |
| Tomato___Leaf_Mold | 952 |
| Cherry_(including_sour)___healthy | 838 |
| Apple___Apple_scab | 630 |
| Apple___Black_rot | 618 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 513 |
| Strawberry___healthy | 454 |
| Grape___healthy | 422 |
| Tomato___Tomato_mosaic_virus | 371 |
| Peach___healthy | 360 |
| Raspberry___healthy | 354 |
| Apple___Cedar_apple_rust | 275 |
| Potato___healthy | 152 |
