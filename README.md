# Resistor Detection Project

## Description
A pure OpenCV computer vision tool that detects electronic resistors in real time using a phone camera as a video feed. Built without machine learning — it uses color segmentation, geometric analysis, and intensity profile peak detection to locate and validate resistors in a scene. Developed as a self-taught project exploring classical computer vision techniques.

## How does it work?
The detection pipeline runs in five stages:
1. Frame preprocessing — resize, Gaussian blur, and histogram equalization on the V channel of HSV to normalize lighting
2. Color segmentation — HSV masks isolate the body colors of common resistors (green, blue, yellow)
3. Morphological closing — fills gaps in the mask and removes noise
4. Geometric validation — minAreaRect fits a rotated bounding box; only shapes with the right aspect ratio, area, and solidity pass
5. Content validation — the ROI is cropped, straightened, and analyzed using a 1D intensity profile. find_peaks detects color bands; peaks are filtered by width, position, and spacing to confirm it's actually a resistor


## What are the results?
The following GIF's show the detector running, containing mixed electronic components placed on a light background.

Blue rotated bounding boxes mark resistors confirmed by the full pipeline: 
color segmentation, geometric validation, and intensity profile band detection. 
Non-resistor components present in the scene are correctly ignored.

![Image of resistors being detected](readme_imgs/1.gif)
![Image of resistors being detected](readme_imgs/2.git)
![Image of resistors being detected](readme_imgs/3.gif)
![Image of resistors being detected](readme_imgs/4.gif)

> **Note:** The detector is currently optimized for light, low-texture 
> backgrounds. Performance may degrade under uneven lighting or on 
> textured surfaces.

## Current Limitations

**Controlled lighting required**
The detector was tested using the phone's flash as a light source. 
This provides consistent, even illumination that stabilizes HSV color 
readings. Performance under ambient or uneven lighting has not been 
validated and may produce missed detections.

**Light, low-texture background**
The algorithm is optimized for operation on a plain light background. 
Textured or cluttered surfaces introduce noise that can interfere with 
color segmentation.

**Hand-tuned color ranges**
The HSV masks were calibrated for a specific camera and lighting setup. 
A different device or environment may require retuning these values.

**Limited resistor body colors**
Currently detects green, blue, and yellow/tan resistor bodies. 
Other common body colors (red, gray) are not supported.

**Detection only — no value reading**
The pipeline confirms the presence of a resistor and counts its color 
bands, but does not yet decode the band colors to output a resistance value.

## Future Roadmap
This project was built using classical computer vision techniques to master image segmentation fundamentals. My next phase is to train a CNN to replace the hand-tuned HSV masks, allowing for better robustness under varying lighting conditions.