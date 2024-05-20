# EEG_ICA_EmotionRecognition
Project based on ICA, CCNN, GCNN for emotion recognition.
This project aims to explore the field of electroencephalographic (EEG) signals in the context of training neural networks for emotion recognition using the SEED IV dataset.

Throughout this study, various methodologies and approaches have been explored in training neural networks for emotion recognition. Among these, particular emphasis has been placed on the use of Independent Component Analysis (ICA) as an analytical tool for preprocessing EEG signals. This technique allows for the separation of signals from different neural sources, enabling a more accurate and detailed analysis of the data.
During this study, in addition to applying Independent Component Analysis (ICA) for preprocessing EEG signals, various approaches in training neural networks for emotion recognition have been explored. In particular, a Continuos Convolutional Neural Network (CCNN) and a Dynamic Graph Convolutional Neural Network (DGCNN) have been utilized. These two models are based on different approaches in data analysis.

## Usage

To utilize this code, please follow these steps:

1. **Download SEED-IV Dataset**: Before running the code, download the SEED-IV dataset from https://www.kaggle.com/datasets/phhasian0710/seed-iv.
  
2. **Install Dependencies**: Ensure you have the following dependencies installed in your Python environment:

   - torcheeg version 1.0.11
   - tensorboard version 2.13
   - rich (latest version)
   - numpy (latest version)
   - sklearn (latest version)
   - typing (latest version)

3. **Run the Code**: Once the dataset is downloaded and dependencies are installed, you can run the code using your preferred Python environment.
```python
python cnn_ica.py
```
```python
python DGCNN_ICA.py
```

3. **Run the Tensorboard**: 
```python
tensorboard --logdir=path/of/training/directory --bind_all 
```
If you want to use a specific port, add the flag --port followed by the desired port number.

## RESULTS
   
While the results of my experimental thesis suggest that the use of ICA does not lead to significant improvements in emotion recognition performance through neural networks, it is crucial to further delve into this analysis by considering other models and techniques. One option to consider could be exploring other types of neural network models beyond the Convolutional Neural Networks (CNN) and Dynamic Graph Convolutional Neural Networks (DGCNN) used in my research. It is possible that other types of neural networks, such as Recurrent Neural Networks (RNN) or Transformer-based models, might achieve better results when combined with the use of ICA. Additionally, adopting the topographic Independent Component Analysis (tICA) approach could provide further improvements in emotion recognition through EEG. tICA is a technique that considers the topographical information of independent components, allowing for better interpretation and selection of relevant components for emotion recognition.
