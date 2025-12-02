import cv2
import numpy as np
import time
from haze_estimation import HazeEstimator
from aqi_classifier import AQIModel

def main():
    # --- CONFIGURATION ---
    # Set to 0 for Webcam, or put a path like 'video.mp4'
    SOURCE = 0 
    
    # Initialize Modules
    estimator = HazeEstimator()
    ai_model = AQIModel()
    
    cap = cv2.VideoCapture(SOURCE)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("System Started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))
        
        # 1. OPTICAL ANALYSIS
        # Get Dark Channel (Visualizing where the haze is)
        dark_channel = estimator.get_dark_channel(frame_resized)
        
        # Estimate Atmospheric Light
        A = estimator.estimate_atmospheric_light(frame_resized, dark_channel)
        
        # Calculate Transmission Map (The "Clarity" of the air)
        transmission = estimator.get_transmission_map(frame_resized, A)
        
        # Calculate Raw Pollution Score (0-100)
        haze_score = estimator.calculate_haze_score(transmission)

        # 2. AI INFERENCE
        # Pass the image and the score to get the Label and PM2.5 estimate
        aqi_label, pm25_est = ai_model.predict_aqi(frame_resized, haze_score)

        # 3. VISUALIZATION
        # Convert transmission to 0-255 grayscale for display
        transmission_display = (transmission * 255).astype(np.uint8)
        transmission_color = cv2.applyColorMap(transmission_display, cv2.COLORMAP_JET)

        # Create Dashboard Overlay
        display = frame_resized.copy()
        
        # Dashboard parameters
        color_status = (0, 255, 0) # Green
        if aqi_label == "Moderate": color_status = (0, 255, 255) # Yellow
        elif aqi_label == "Unhealthy": color_status = (0, 165, 255) # Orange
        elif aqi_label == "Hazardous": color_status = (0, 0, 255) # Red

        # Draw UI
        cv2.rectangle(display, (0, 0), (640, 80), (30, 30, 30), -1)
        cv2.putText(display, f"AQI Status: {aqi_label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
        cv2.putText(display, f"Est PM2.5: {pm25_est:.1f} ug/m3", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f"Haze Density: {haze_score:.1f}%", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Show the "AI Vision" (Transmission Map) as a picture-in-picture
        # Resize transmission map to small box
        pip_h, pip_w = 120, 160
        transmission_small = cv2.resize(transmission_color, (pip_w, pip_h))
        
        # Overlay in bottom right
        h, w, _ = display.shape
        display[h-pip_h-10:h-10, w-pip_w-10:w-10] = transmission_small
        cv2.rectangle(display, (w-pip_w-10, h-pip_h-10), (w-10, h-10), (255, 255, 255), 1)
        cv2.putText(display, "Pollution Map", (w-pip_w-5, h-pip_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        cv2.imshow('Urban Air Quality Assessment', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()