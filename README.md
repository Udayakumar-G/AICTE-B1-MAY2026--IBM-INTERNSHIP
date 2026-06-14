# AICTE-B1-MAY2026--IBM-INTERNSHIP
AI BASED AUTOMATED DISCIPLINE CHECKING SYSTEM
An AI-powered system designed to automate dress code verification, grooming evaluation, posture check, and ID card detection in real-time. Built using Deep Learning (MobileNetV2) and Computer Vision (OpenCV), this project replaces inefficient manual monitoring with automated, data-driven discipline insights.
🎯 Objectives
To automate dress code verification within classrooms or institutional premises.
To drastically reduce manual monitoring efforts, freeing up valuable staff time.
To achieve high-accuracy attire classification using Transfer Learning (MobileNetV2).
To engineer a seamless, live visual interface utilizing TensorFlow and OpenCV.
🧠 Problem Statement vs. Proposed Solution
The Existing System & DisadvantagesManual Staff Monitoring: High deployment costs, prone to human error, fatigue, bias, and congestion during peak hours.
Static CCTV Surveillance: Captures basic footage but fails to automatically identify dress code violations or maintain a log of repeat offenders.
Our Proposed AI SolutionAutomated & Instantaneous: An AI-backed platform that checks student attire compliance 10\times faster than human checks.
Cost-Efficient & Scalable: Cuts down monitoring workload and operational costs by up to 70%, with direct scalability to schools, factories, and defense complexes.
Data-Driven Dashboards: Logs analytics natively to map policy improvements and trend visualizations.
🏗️ System Architecture & Modular Breakdown
1. Data Preprocessing ModuleNormalizes, augments (rotation, flipping, brightness adjustments), and resizes input images to 224 \times 224 pixels to feed cleanly into the network backbone.
2. 2. Model Training ModuleUtilizes MobileNetV2 (pre-trained on ImageNet) as a feature extraction foundation, appending a customized classification layer outfitted with Softmax activation to predict dress states.
3. Validation ModuleTests the system against isolated verification datasets to optimize accuracy, balance loss metrics, and tune hyper-parameters to bypass overfitting pitfalls.
4. Real-Time Detection ModuleIntegrates an OpenCV video stream interface to instantly predict student parameters frame-by-frame and cast overlay data labels onto the screen.
5. Result Display & User Interface ModuleHouses an interactive control panel featuring standard Start/Stop execution controls, error/success warnings, and clear visual readouts of compliance fields.
📊 Monitored Parameters & Metrics
The application monitors and segments compliance across multiple operational parameters:Dress Code: Formal vs. Casual/InformalGrooming (Beard): Clean Shaven vs. BeardPosture: Tucked In (Tuckin) vs. Not Tucked In (Nottuckin)ID Card Presence: Wearing vs. Not Wearing
🖥️ Dashboard Features
The integrated AI Discipline Detection Dashboard comes packed with deep analytical tracking components:
Overview Metrics: Displays total active detections, unique tracked persons, and compliance percentages.
Trend Analysis: Graphs daily compliance trends alongside detection tracking distribution by hour.Compliance Breakdown: Offers clear pie-chart breakdowns tracking grooming, posture distributions, and ID card violations.
Data Exports: Built-in options to export collected data via CSV summaries, text reports, or ZIP compressed images.
