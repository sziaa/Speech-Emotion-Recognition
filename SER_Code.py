#Code Source: https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/#goog_rewarded
#Install necessary packages
!pip install librosa 
!pip install soundfile 
!pip install numpy 
!pip install scikit-learn 
!brew install portaudio
!pip install pyaudio

#Import packages:

#analyzes audio and music files
import librosa

#reading and writing sound files
import soundfile

#os allows for interacting with operating systems; glob allows for searching for pathname patterns; pickle allows for serializing and deserializing Python objects
import os, glob, pickle

#numerical calcutions
import numpy as np

#Splits model into data sets to train and test (ML)
from sklearn.model_selection import train_test_split

#multi layer percpton classifier -> type of feedforward neural network used for learning tasks
from sklearn.neural_network import MLPClassifier

#computes accuracy of the classified model
from sklearn.metrics import accuracy_score

#Create a function extract to pull feature data from a sound file. 
#Inputs of file path and 3 boolean values of mfcc, chroma, mel
      #MFCC: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound (timbral features reflecting how humans percive frequencies)
      #Chroma: 12 different pitch classes of music (C,C#,D,D#..) -> tonal component
      #Mel: Mel Spectrogram Frequency, power of the signal across different frequencies (Visualizes the energy distribution across frequencies on a mel scale)
#Ouputs: if any of the 3 booleans are true extract the mean value of it

def extract(file_name, mfcc, chroma, mel):
    #open soundfile
    with soundfile.SoundFile(file_name) as sound_file:

        #read audio data into an array and convert to float32 type
        X = sound_file.read(dtype = "float32")

        #get sample rate (how frequently the analog audio signal is sampled converted into digital form)
        sample_rate = sound_file.samplerate

        #initalize an empty array to store results (mean values of 3 features outlined above)
        result = np.array([])

        #if chroma feature extracted is enable, compute short time fourier transformation of the audio data
        # STFT  maps a signal into a two-dimensional function of time and frequency
        if chroma: 
            stft = np.abs(librosa.stft(X))

        # if mfcc is true, compute the mfcc's and take the mean. horizontally stack the result into result array
        if mfcc: 
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            results = np.hstack((result,mfccs))


        #if chroma is true, compute chroma features from stft and take mean. horizontally stack the result into result array
        if chroma: 
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis = 0)
            result = np.hstack((result, chroma))

        #if mel is true, compute mel spectrogram from data and take mean. horizontally stack the result into result array
        if mel: 
            mel = np.mean(librosa.feature.melspectrogram(y= X, sr=sample_rate).T,axis = 0)
            result = np.hstack((result, mel))

    return result

#comprehensive dictonary of all emotions detectable in the data set:
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'}

#list of emotions we are looking to observe:
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#load in data and extract features for each sound file depending on if that file has an emotion we want to observe indicated thorught the dictonary defined above
#Input: relative size of the test size
def load_data(test_size=0.2):
    
    #initalize empty lists for features (x) and lables (y) of the sound files we want to consider
    x, y = [],[]


    #loop through all WAV files in directory using glob (relative path in my case)
    for file in glob.glob("RAVDESS/Actor_*/*.wav"):
        #get file name
        file_name = os.path.basename(file)
    
        #Extract the emotion label from the file name (assuming the corresponding emotion is in the file name)
        emotion = emotions[file_name.split("-")[2]]
    
        #if the file does not include an emotion we want to observe (observed_emotions list), skip file
        if emotion not in observed_emotions:
            continue
    
        #OW extract mfcc, chroma and mel features from sound file 
        feature = extract(file, mfcc = True, chroma= True, mel = True)
    
        #Append the extracted features to the x list
        x.append(feature)
        
        #Append the correpsonding emotion to the y list
        y.append(emotion)

    #split data into training and testing test for machine learning
    return train_test_split(np.array(x), y, test_size = test_size, random_state = 9)

#Split data into sets for training and testing; we will use 25% of data for testing and 75% for training
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

# Observe the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# Observe the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#Define the MLP Classifier that optimizes the log-loss function
    #Alpha: L2 regulariztion term that prevents overfitting by penalizin large weights in the model (larger alphja has stronger regularization)
    #batchsize: number of training samples used in one iteration
    #epsilon: added to denominator of graident descent to prevent divison of 0
    #hidden_layer_sizes = hidden layer architecture (gere we have 1 hiden layer w 300 neaurons)
    #learning_rate: adaptive means the rate decreases only when model stops improving
    #max_iter: maximum epcohs(number of iterations) for model training 
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=800)


#Train model to learn patterns:
model.fit(x_train,y_train)

#Predict the values for the test sets (using x_test)
y_pred=model.predict(x_test)

#Calculate accuracy of the model by comparing y_pred and y_test
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#print result
print("Accuracy: {:.2f}%".format(accuracy*100))

# Function to preprocess user-uploaded audio files; ie pull out mcff, chroma, mel
def preprocess_user_audio(file_path):
    
    # Extract features from the audio file using the same method as in training
    feature = extract(file_path, mfcc=True, chroma=True, mel=True)
    
    # Return the features in the correct shape for model input
    return np.array([feature])

# Function to predict emotion from an audio file
def predict_emotion(file_path):
    
    # Preprocess the audio file to extract features
    features = preprocess_user_audio(file_path)
    
    # Use the trained model to predict the emotion (outputs a list of probabilities)
    #emotion_probabilities = model.predict(features)
    emotion_probabilities = model.predict_proba(features)[0]

    #print(emotion_probabilities)

    # Find the index of the highest probability
    highest_probable_index = np.argmax(emotion_probabilities)

    # Convert the index to the corresponding emotion label using the 'emotions' dictionary
    predicted_emotion = emotions[list(emotions.keys())[highest_probable_index]]

    # Return the predicted emotion
    return predicted_emotion

# Command-line interface (CLI) for user interaction
if __name__ == "__main__":
    
    # Prompt the user to enter the path to an audio file
    file_path = input("Enter the path to the audio file: ")
    
    # Predict the emotion from the provided audio file
    emotion = predict_emotion(file_path)
    
    # Print the predicted emotion
    print(f"Predicted emotion: {emotion}")