import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '

import streamlit as st
from pydub import AudioSegment
from audioProcessing import*
from gradCam import*


st.title("Grad-CAM Activation for Audio using 2DCNN")

st.markdown('''
We have trained our model in order to find if a Capuchin Bird call is present or not in the audio file. This is a binary classification.
1 represent bird call is present 0 represent the absence of the same. We have also provided some exapmle files for each of the classification. Choose one!! :bird: 
''')
# Define the path to the folder containing the audio files
AUDIO_FOLDER = "data"



# Create a slider for selecting the number
number = st.slider("Select a number", 0, 1)

# Get the path to thess directory containing the audio files for the selected number
audio_dir = os.path.join(AUDIO_FOLDER, str(number))

# print(audio_dir)
# Get the names of all audio files in the directory

if 'audio_file' not in st.session_state:
    
    st.session_state.audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]


# print(audio_files)

# Create a dropdown for selecting the audio file
file_name = st.selectbox("Select an audio file", st.session_state.audio_files)


# Read the audio file using PyDub
audio_path = os.path.join(audio_dir, file_name)
#print(audio_path)
audio_file = AudioSegment.from_file(audio_path)

# Convert the audio file to a playable format and display it in the app
audio_bytes = audio_file.export(format="wav").read()
st.audio(audio_bytes, format="audio/wav")

wave = load_wav_16k_mono(audio_path)
spectrogram, _ = preprocess(audio_path, number)
fig, ax = plt.subplots(1,2) 
ax[0].plot(wave)
ax[0].set_title('Image 1')
ax[0].text(0.5,-0.1,'Audio file in waveform',transform=ax[0].transAxes,ha ='center',va='center')
ax[1].imshow(spectrogram)
ax[1].set_title('Image 2')
ax[1].text(0.5,-0.1,'Spectrogram of our wave form',transform=ax[1].transAxes,ha ='center',va='center')
st.pyplot(fig)
but = st.button('Predict')

if but:
    print("Prediction initiated")
    prediction = predict(audio_path=audio_path,number=number)
    if (prediction):
        st.write("Capuchin Bird present")  
    else:
        st.write("Capuchin Bird not Present")
    st.image('cam.jpg')
    