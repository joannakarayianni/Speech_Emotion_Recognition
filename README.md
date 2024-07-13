# Speech Emotion Recognition
• Voluntary project for the course Introduction to Deep Learning for Speech and Language Processing

• Classification problem where we needed to predict the emotion class based on speech input features.

• There are 4 classes where we used a two-dimensional representation:


<img width="1440" alt="Στιγμιότυπο οθόνης 2024-07-13, 8 30 20 μμ" src="https://github.com/user-attachments/assets/63192d86-0724-4eb6-9da8-00d91aab3b2f">


• I used a Bi-LSTM (Bidirectional Long Short-Term Memory) model for the emotion classification. The model is trained to predict valence and activation states from the given features. The datasets used includes training, validation, and testing splits in JSON format.

# Installation 
``` git clone https://github.com/joannakaragianni/speech_emotion_recognition.git ```

``` cd speech_emotion_recognition ```

• Create virtual enviroment:

``` python -m venv venv ```

``` source venv/bin/activate ```

• Install requirements:

``` pip install -r requirements.txt ```

• Run file:

``` python3 ./scripts/emotion_analysis_BiLSTM_CE.py ```
