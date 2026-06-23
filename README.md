# Resistor Detection Project

## Description
An OpenCV-based computer vision tool designed to identify electronic resistors in real-time. This project uses a combination of HSV color-space masking, morphological transformations, and geometric validation to isolate resistors.

## Results
The algorithm currently achieves high detection accuracy on solid white backgrounds. Below are examples of the system successfully isolating resistors from other components:

![Image of resistors being detected](imagenes/github_README/Picture2.png)
![Image of resistors being detected](imagenes/github_README/Picture1.png)
![Image of resistors being detected](imagenes/github_README/Picture3.png)

## Current Limitations
**Background Sensitivity:** The detection algorithm is optimized for operation on a solid white background. Using textured or varied backgrounds may introduce noise and false positives.
