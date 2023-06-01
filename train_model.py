import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '

from audioProcessing import*
import tensorflow as tf 




print("Importing Data")
POS = os.path.join('data_original', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('data_original', 'Parsed_Not_Capuchinbird_Clips')

pos = tf.data.Dataset.list_files(POS+'\*.wav')
neg = tf.data.Dataset.list_files(NEG+'\*.wav')

nene = tf.zeros(len(neg))


print((neg, tf.data.Dataset.from_tensor_slices(nene)))
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(nene)))
data = positives.concatenate(negatives)


print("Preparing Data")
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

print("Partitioning Data")
print(len(data))
train = data.take(36)
test = data.skip(36).take(15)

samples, labels = train.as_numpy_iterator().next()
print(labels.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding = "same",input_shape=(1491,257,1),name ="conv2d_1"))
model.add(MaxPooling2D(2,2,name="maxpooling_1"))
model.add(Conv2D(64,(3,3),activation='relu',padding = "same",name = "conv2d_2"))
model.add(MaxPooling2D(2,2,name="maxpooling_2"))
model.add(Flatten())
model.add(Dense(128,activation="relu", name = "dense_1"))
model.add(Dense(1,activation="sigmoid",name="dense_2"))
print("Layers Added....now compiling")
model.compile('Adam',loss='BinaryCrossentropy',metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),tf.keras.metrics.Accuracy()])
model.summary()

print("Fitting the model")
hist = model.fit(train, epochs=4,validation_data=test,verbose=1)
model.save('finalModel.h5')


