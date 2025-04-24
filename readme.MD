# Microchip Detection and Reporting System

This project is a comprehensive system for detecting microchip defects, analyzing data, and generating reports. It integrates machine learning models, a graphical user interface (GUI), and database connectivity to streamline the process of defect detection and reporting.

## Features

- **Defect Detection**: Uses CNN models to classify microchip defects in grayscale and RGB images.
- **Data Analysis**: Leverages LLMs (Large Language Models) for SQL query generation and result analysis.
- **GUI**: A PyQt-based interface for managing detection workflows, visualizing data, and exporting results.
- **Database Integration**: Supports MySQL for storing and querying defect data.
- **Export Options**: Allows exporting data to Excel and CSV formats.
- **Camera Integration**: Supports live camera feeds for real-time defect detection.

## Project Structure

- **Detection**: Contains CNN models and scripts for training, classification, and preprocessing.
  - `CNN_code`: Includes scripts for training and classifying grayscale and RGB images.
  - `labelImg_xml_and_crop`: Scripts for cropping and processing images based on XML annotations.
- **GUI**: PyQt-based graphical interface for managing the detection process.
  - `main.py`: Main entry point for the GUI.
  - `Uncompiled`: Contains additional GUI components and utilities.
- **LLM**: Scripts for integrating large language models for SQL generation and analysis.
  - `chatbotGPT.py`: Handles chatbot interactions and SQL query generation.
  - `gemini.py` and `tinyllama.py`: Alternative LLM integrations.
- **ESP32_Project**: Arduino-based project for hardware integration.

## Requirements

- Python 3.8+
- TensorFlow
- PyQt5 or PyQt6
- MySQL
- OpenCV
- Ultralyitcs YOLO
- Google Generative AI (optional for LLM integration)
- CUDA 12+

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/microchip-detection.git
   cd microchip-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database:

   - Configure MySQL with the credentials in `chatbotGPT.py`.
   - Create the required tables using the provided schema.

4. Run the GUI:
   ```bash
   python GUI/main.py
   ```

## Usage

1. Launch the GUI and load the trained CNN model.
2. Use the camera feed or upload images for defect detection.
3. View and analyze results in the data table.
4. Export results to Excel or CSV.
5. Use the chatbot for SQL-based data queries and analysis.

## Video Demonstration

[![Watch the video](vIDEO.mp4)](vIDEO.mp4)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- TensorFlow for deep learning.
- PyQt for GUI development.
- MySQL for database management.
- Google Generative AI for LLM integration.
