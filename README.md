AgriListen: Audio-to-Vision Autoencoders with LLM Decision Support
An end-to-end Deep Learning and Generative AI ecosystem designed to detect early-stage internal infestations of the Red Palm Weevil (RPW) in coconut palms. By leveraging acoustic telemetry, this project bridges Computer Vision (audio-to-spectrogram conversion), Unsupervised Anomaly Detection, and a Retrieval-Augmented Generation (RAG) Decision Support System to provide an actionable, green-tech agricultural solution.

Overview
The Red Palm Weevil destroys palm trees from the inside out, often causing irreversible structural damage before visual symptoms appear. While acoustic monitoring can "hear" the larvae chewing, acquiring extensive training datasets of infested trees is extremely difficult.

AgriListen solves this data scarcity challenge by training a Convolutional Autoencoder strictly on the acoustic signatures of healthy palm trunks. When exposed to the chewing transients of an infested tree, the model fails to reconstruct the high-frequency pulses, triggering an anomaly alert. The system is coupled with a Groq LLM RAG Pipeline that synthesizes the quantitative Deep Learning loss metrics with retrieved agronomic rules to generate a localized, explainable action plan for farmers.

Core Features
Audio-to-Vision Pipeline: Converts raw .wav telemetry into Mel-spectrogram images using librosa to leverage powerful CNN spatial feature extraction.

Semi-Supervised Autoencoder: Detects anomalies without requiring prior exposure to infested data, establishing a robust baseline threshold for environmental ambient noise.

Geospatial Farm Mapping: An interactive Streamlit dashboard mapping sensor nodes to physical tree coordinates (simulated in Nagercoil, TN). Updates tree status (Healthy/Infested) dynamically upon inference.

Explainable AI (Saliency Heatmaps): Visualizes the exact temporal-frequency coordinates where the model failed reconstruction, mathematically isolating the localized chewing transients.

GAN Synthetic Data Simulation: A generative module demonstrating how synthetic bioacoustic anomalies can be generated to balance sparse agricultural datasets.

Context-Aware AI Agronomist: A RAG pipeline utilizing the Groq API (openai/gpt-oss-20b) to translate Mean Squared Error (MSE) metrics into explicit, step-by-step agricultural protocols.

Tech Stack
Deep Learning Framework: PyTorch, Torchvision

Audio Processing: Librosa, SciPy

Data & Math: NumPy, Pandas, Matplotlib

Generative AI / LLM: Groq API

Web UI / Edge Simulation: Streamlit

Installation & Setup
1. Clone the repository:

Bash
git clone https://github.com/yourusername/agrilisten.git
cd agrilisten
2. Install dependencies:
(Python 3.12 recommended)

Bash
pip install torch torchvision torchaudio librosa matplotlib numpy scipy pandas streamlit groq
3. Configure the Groq API Key:

Obtain a free API key from Groq Console.

You can paste it directly into the Streamlit UI sidebar at runtime, or hardcode it into app.py by updating the HARDCODED_GROQ_KEY variable.

4. Generate the Synthetic Dataset and Train the Baseline Model:

Bash
python generate_dataset.py
python train.py
(This generates synthetic ambient healthy audio and trains the Convolutional Autoencoder, saving the baseline weights to rpw_autoencoder.pth.)

5. Launch the Dashboard:

Bash
streamlit run app.py
Usage Workflow
Open the Streamlit dashboard (http://localhost:8501).

Select a Tree ID from the interactive Farm Setup sidebar.

Upload an acoustic sample (.wav) or record live via the microphone input.

Review the Autoencoder reconstruction visualizations, anomaly saliency heatmap, and real-time Geospatial map updates.

Click Generate Field Action Protocol to trigger the RAG pipeline and receive a customized agronomic advisory report.

Academic Mapping & Research Context
This architecture actively maps to advanced deep learning curriculum concepts and is structured as a foundational prototype for Agricultural IoT research:

Unit I & II (Feature Extraction & CNNs): Utilizes Convolutional layers and Multi-Layer Perceptrons for spatial mapping of temporal audio frequencies.

Unit III (Autoencoders & Generative AI): Applies dimensionality reduction, noise filtering for semi-supervised classification, and explores synthetic data augmentation (GAN concepts) for imbalanced datasets.

Transfer Learning & Deployment: Deploys trained .pth weights into an interactive web simulation for edge-device feasibility testing.
