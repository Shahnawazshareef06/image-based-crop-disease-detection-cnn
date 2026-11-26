# TerraVista | Deployment brief for agricultural extension staff

**Use TerraVista to screen a leaf photo and decide what to inspect next.** Ask the grower which crop is affected, where symptoms first appeared, and whether other plants show the same pattern. Keep that history alongside the photo when reviewing the model’s suggestion. [Open the leaf classifier](https://image-based-crop-disease-detection-cnn.streamlit.app/)

### Which model should the service use?

| Model | Size for 38 classes | Role at the extension desk |
|---|---|---|
| **MobileNetV3-Large** | 4.25 million parameters; **17.21 MB** saved weights | Start here. The smaller download suits a CPU laptop or a connection with limited bandwidth. |
| **ResNet-50** | 23.59 million parameters; **about 94 MB** of FP32 parameters | Use for a second assessment. It scored higher on both lab and field photos, but agreement between models still needs field confirmation. |

The app downloads the selected seed-42 checkpoint once and caches it. For an offline visit, run one prediction on the laptop with each model you plan to use before leaving. The MobileNet size is measured on disk. The ResNet figure is an estimate from parameter count × 4 bytes; the saved file also includes buffers and metadata. [MobileNet training record](../models/mobilenetv3_large/results/mobilenetv3/mobilenetv3_large_uniform_p010_seed42/metrics.json) · [Offline setup](../README.md#get-the-trained-weights)

### How to handle a grower’s photo

1. **Check coverage first.** The 38 labels are crop–condition combinations across 14 crops: apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry and tomato. Even a listed crop can have an unlisted disease. For rice, wheat, an unfamiliar crop, or symptoms outside the class list, record **outside scope** and use the usual extension diagnostic process. The app can give an unsupported input a high score. [Full class list](../models/mobilenetv3_large/class_to_idx.json)
2. **Below 80% calibrated confidence → Unsure.** Ask for a sharp close-up of one affected leaf in even light against a plain background. If the result stays unsure or the predicted crop conflicts with the grower’s identification, refer the case to a crop specialist.
3. **At or above 80% → review the suggested class.** Compare it with symptoms across the plant. The orange Grad-CAM overlay shows which image regions influenced the score and may highlight background. Confirm the cause before using the class-specific care guidance or recommending treatment.

### What the field test showed

On the same 27 classes, **ResNet-50 scored 0.981 lab / 0.282 field macro-F1** and **MobileNetV3-Large scored 0.961 / 0.232**. These are three-seed averages on 5,169 screened PlantVillage images and 241 PlantDoc field images. Macro-F1 weights each class equally. The 80% cutoff was calibrated on lab leaves; an 80% score on a farm photo does not establish an 80% chance of a correct diagnosis. [Field evaluation](../experiments/plantdoc-field-test/README.md) · [Calibration study](calibration/README.md)
