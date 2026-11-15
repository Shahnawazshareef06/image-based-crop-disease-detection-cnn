# ResNet-50 vs. MobileNetV3: Model Comparison & Error Analysis

A clean, direct comparison and error analysis of **ResNet-50** and **MobileNetV3-Large** evaluated on the 38-class PlantVillage dataset across seeds `42`, `123`, and `2026`.

---

## 1. Overall Model Comparison

| Metric | ResNet-50 | MobileNetV3-Large | Delta / Difference |
| :--- | :---: | :---: | :---: |
| **Validation Macro-F1** | **0.9847 ± 0.0009** | **0.9685 ± 0.0015** | ResNet-50 (+0.0162, p = 0.0015) |
| **Validation Accuracy** | **98.62%** | **97.49%** | ResNet-50 (+1.13%) |
| **Parameters** | 25.56 Million | **5.48 Million** | MobileNetV3 is **4.6x smaller** |
| **Model Size** | 97.8 MB | **21.4 MB** | MobileNetV3 is **4.6x lighter** |
| **Latency (ms/image)** | 12.4 ms | **3.8 ms** | MobileNetV3 is **3.3x faster** |
| **Throughput (img/sec)** | 80.6 img/s | **263.2 img/s** | MobileNetV3 is **3.3x higher** |
| **Best Loss** | CrossEntropy | Focal Loss ($\gamma=1.0$) | Task-dependent |

![Model Comparison Overview](figures/model_comparison_overview.png)

---

## 2. Individual Weakest Classes

### ResNet-50: Individual Weakest Classes
| Rank | Class Name | F1-Score | Precision | Recall | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `Tomato___Early_blight` | **0.8715** | 0.8841 | 0.8592 | 142 |
| 2 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | **0.8851** | 0.9199 | 0.8571 | 77 |
| 3 | `Tomato___Target_Spot` | **0.8924** | 0.9015 | 0.8836 | 219 |
| 4 | `Tomato___Late_blight` | **0.9328** | 0.9410 | 0.9248 | 295 |
| 5 | `Corn_(maize)___Northern_Leaf_Blight` | **0.9432** | 0.9290 | 0.9592 | 147 |
| 6 | `Potato___healthy` | **0.9654** | 0.9733 | 0.9583 | 24 |

### MobileNetV3: Individual Weakest Classes
| Rank | Class Name | F1-Score | ResNet-50 F1 | Support |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `Tomato___Early_blight` | **0.8467** | 0.8715 | 142 |
| 2 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | **0.8563** | 0.8851 | 77 |
| 3 | `Tomato___Target_Spot` | **0.8712** | 0.8924 | 219 |
| 4 | `Tomato___Late_blight` | **0.9255** | 0.9328 | 295 |
| 5 | `Potato___healthy` | **0.9285** | 0.9654 | 24 |
| 6 | `Corn_(maize)___Northern_Leaf_Blight` | **0.9351** | 0.9432 | 147 |

![Individual Weakest Classes](figures/individual_weakest_classes.png)

---

## 3. Individual Most Confused Disease Pairs

### ResNet-50: Most Confused Pairs
| Rank | Class A | Class B | Errors | Total Support | Error Rate |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `Tomato___Early_blight` | `Tomato___Target_Spot` | **8** | 361 | 2.22% |
| 2 | `Corn___Cercospora_leaf_spot` | `Corn___Northern_Leaf_Blight` | **7** | 224 | 3.12% |
| 3 | `Tomato___Early_blight` | `Tomato___Late_blight` | **6** | 437 | 1.37% |
| 4 | `Grape___Black_rot` | `Grape___Leaf_blight` | **5** | 340 | 1.47% |
| 5 | `Apple___Apple_scab` | `Apple___Black_rot` | **3** | 189 | 1.59% |
| 6 | `Pepper,_bell___Bacterial_spot` | `Tomato___Bacterial_spot` | **2** | 470 | 0.43% |

### MobileNetV3: Most Confused Pairs
| Rank | Class A | Class B | Errors | Total Support | Error Rate |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `Tomato___Early_blight` | `Tomato___Target_Spot` | **18** | 361 | 4.99% |
| 2 | `Tomato___Early_blight` | `Tomato___Late_blight` | **16** | 437 | 3.66% |
| 3 | `Corn___Cercospora_leaf_spot` | `Corn___Northern_Leaf_Blight` | **14** | 224 | 6.25% |
| 4 | `Grape___Black_rot` | `Grape___Leaf_blight` | **11** | 340 | 3.24% |
| 5 | `Apple___Apple_scab` | `Apple___Black_rot` | **9** | 189 | 4.76% |
| 6 | `Pepper,_bell___Bacterial_spot` | `Tomato___Bacterial_spot` | **7** | 470 | 1.49% |

![Individual Most Confused Pairs](figures/individual_most_confused_pairs.png)

---

## 4. Multi-Sheet Excel Workbook: `resnet50_vs_mobilenetv3_comparison_and_error_analysis.xlsx`

Contains **10 clean data sheets** :
1. **`Model_Comparison_Summary`**: Clean side-by-side specs and metrics.
2. **`Per_Seed_Results`**: Exact scores for seeds 42, 123, 2026.
3. **`All_38_Classes_Comparison`**: Full per-class F1, Precision, Recall, and Delta.
4. **`ResNet50_Weakest_Classes`**: Ranked weakest classes for ResNet-50.
5. **`MobileNetV3_Weakest_Classes`**: Ranked weakest classes for MobileNetV3.
6. **`ResNet50_Confused_Pairs`**: Individual most confused pairs for ResNet-50.
7. **`MobileNetV3_Confused_Pairs`**: Individual most confused pairs for MobileNetV3.
8. **`Confused_Pairs_Comparison`**: Side-by-side error count comparison.
9. **`ResNet50_Confusion_Matrix`**: 38x38 numerical confusion matrix.
10. **`MobileNetV3_Confusion_Matrix`**: 38x38 numerical confusion matrix.

---

