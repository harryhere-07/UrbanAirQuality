@echo off
echo Starting Air Quality System...
pip install opencv-python numpy tensorflow streamlit > nul
streamlit run webapp.py
pause