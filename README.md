# Real-Time Urban Air Quality Assessment Using Computer Vision

## Project Overview
This system utilizes Computer Vision and Deep Learning concepts to estimate urban air quality without physical sensors. By analyzing visual features such as **haze intensity**, **atmospheric light**, and **transmission maps** from camera feeds, the system infers PM2.5 levels and classifies air quality.

## Features
- **Dark Channel Prior (DCP) Algorithm:** Extracts the "Transmission Map" of an image to detect suspended particulate matter.
- **Atmospheric Light Estimation:** Auto-calibrates based on the brightest pixels in the scene.
- **Real-Time Dashboard:** Overlay showing AQI Category, Estimated PM2.5, and Haze Density.
- **Heatmap Visualization:** A Picture-in-Picture view showing exactly where the algorithm detects pollution in the frame.

## Technical Architecture
1. **Input:** Surveillance Video or Webcam feed.
2. **Preprocessing:** Gaussian blur and resizing.
3. **Feature Extraction:** - $J(x) = \frac{I(x) - A}{t(x)} + A$
   - Calculation of Dark Channel and Transmission $t(x)$.
4. **Classification:** Mapping optical density to AQI standards (Good/Moderate/Hazardous).

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`
3. Press 'q' to exit the application.