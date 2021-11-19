1. Problem
This benchmark is based on the SSD: Single Shot MultiBox Detector paper, which describes SSD as “a method for detecting objects in images using a single deep neural network”.

2. Directions
Example run:

python run_benchmark.py -b ssd_coco --datadir /mnt/cota/datasets/coco --name ssd  --exp_name ssd-baseline --run_name ssd --namespace laura --instance cota-g1


3. Dataset/Environment
This SSD-ResNet34 model was trained on the COCO 2017 dataset. The val2017 validation set was used as a validation dataset.


4. Model
file: ssd_diagram.png

5. Quality metric
After 65 epochs:  Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = 0.250.