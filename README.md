# EPR
**Overview**
The EPR Analysis Interface is a Python-based graphical user interface (GUI) application designed to streamline electron paramagnetic resonance (EPR) data analysis for researchers and scientists. Built on the tkinter library, this software offers a wide range of functionalities to facilitate the processing, analysis, and visualization of EPR data. It can import data from multiple formats, apply advanced noise reduction and baseline correction techniques, and generate customizable plots tailored to specific research needs. Additionally, it includes a predictive model for identifying radical types from EPR spectrum data with high accuracy.

**Features**
Modular and Extensible Design: Built with scalable class structures, the application is easy to extend with additional features as needed.
Multi-format Data Import: Supports data import from five different EPR data formats, enhancing compatibility with various instruments.

**Data Processing Tools:**
1. Noise reduction and baseline correction for cleaner, more accurate data analysis.
2. Advanced signal processing and visualization tools.
3. Customizable Visualization: Generate and customize plots to suit research requirements.
4. 
**Predictive Model for Radical Identification:**
1. Uses a logistic regression model to predict radical types from EPR spectrum data.
2. Achieves a prediction accuracy of 92.9%.
3. User-friendly Interface: Intuitive design that facilitates efficient data handling and minimizes the learning curve.
   
**Libraries**
1. tkinter for GUI
2. numpy, pandas, matplotlib, scikit-learn for data processing and analysis
3. scipy for baseline correction and noise reduction algorithms

Load EPR Data: Use the "Input Data" button to import data files in supported formats.
Data Preprocessing Options: Apply noise reduction, baseline correction, and other preprocessing options from the interface.
Plotting and Visualization: Customize plots for enhanced visualization and interpretation of your EPR data.
Predict Radical Type: Run the logistic regression model on the loaded spectrum data to predict radical types with high accuracy.
