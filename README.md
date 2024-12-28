# Hindi_Speech_Emotion_Recognition.
This project aims to recognize emotions (happy, sad, fear, neutral, sarcastic, and surprise) from speech data in Hindi using machine learning techniques. The model was trained using speech features extracted from the audio data, and we used **MFCCs** (Mel-Frequency Cepstral Coefficients) for feature extraction.

## Objective
The goal of this project was to:
- Identify emotions from speech data in the Hindi language.
- Use speech signal processing techniques (MFCCs) for feature extraction.
- Build a classification model (machine learning or deep learning) for emotion recognition.
- Evaluate the model's performance using metrics like accuracy and F1-score.

## Dataset
The dataset consists of speech files labeled with different emotions (happy, sad, fear, neutral, sarcastic, and surprise). Each emotion category contains 10 speech files, making it a small, imbalanced dataset.

### Dataset Details
- **Emotions**: Happy, Sad, Fear, Neutral, Sarcastic, Surprise
- **Number of Files**: 60 (10 files for each emotion)
- **Feature Extraction**: MFCCs (13 features per sample)

### Preprocessing
The following preprocessing steps were applied to the dataset:
1. **Audio Preprocessing**: 
   - Speech data was loaded and resampled.
   - **MFCCs** were extracted from each audio file to represent the speech features.
2. **Data Balancing**: 
   - The dataset was imbalanced, so techniques like **SMOTE** were used for data balancing, though the model still struggled with underrepresented classes.

## Model
A **Random Forest Classifier** was used to classify the emotions based on the MFCC features. The model was trained on the processed feature set and evaluated on the test set.

### Model Evaluation
The model was evaluated using accuracy and F1-scores for each class. Here are the results:

### Classification Report:
              precision    recall  f1-score   support

        fear       0.00      0.00      0.00         0
       happy       0.00      0.00      0.00         2
     neutral       0.00      0.00      0.00         2
         sad       0.00      0.00      0.00         2
   sarcastic       1.00      0.33      0.50         3
    surprise       0.33      0.33      0.33         3

    accuracy                           0.17        12
   macro avg       0.22      0.11      0.14        12
weighted avg       0.33      0.17      0.21        12

### Key Observations:
- **Accuracy**: The overall accuracy of the model is 17%. This indicates poor performance.
- **F1-Score**: The F1-scores for most emotions are 0, reflecting that the model struggles with predicting the majority of emotions accurately.
- **Sarcastic Emotion**: The model performs best on the sarcastic emotion but still has low recall and precision.

### Challenges:
- **Imbalanced Data**: The dataset was highly imbalanced, with only 10 files for each emotion. This made it difficult for the model to learn the characteristics of each emotion properly.
- **Small Dataset**: The dataset size was small, which may have contributed to overfitting and poor generalization.
- **Feature Representation**: MFCC features were used for audio representation, but further experimentation with other techniques like spectrograms might improve performance.

## Future Work
1. **Increase Dataset Size**: A larger and more balanced dataset would improve the model's performance.
2. **Advanced Feature Extraction**: Experimenting with advanced feature extraction techniques such as **Chroma features**, **Spectrograms**, or **Deep learning-based features** could provide better representations for emotion recognition.
3. **Model Optimization**: Exploring other models like **Convolutional Neural Networks (CNNs)** or **Recurrent Neural Networks (RNNs)**, which are effective for speech data, could help achieve better results.
4. **Data Augmentation**: Implementing data augmentation techniques like pitch shifting, noise addition, and time stretching might help improve the model’s robustness.

## Conclusion
This project demonstrates an initial attempt to recognize emotions from Hindi speech using traditional machine learning models. Although the results are far from optimal, the project highlights the importance of having a diverse and balanced dataset for emotion recognition tasks. Further improvements can be made by addressing the challenges mentioned above and experimenting with more advanced models and techniques.
