import numpy as np
import tensorflow as tf
from tensorflow import keras
from audioProcessing import*

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Rescale heatmap to a range 0-255
    print("Img shape",img.shape)
    print("Heatmap shape",heatmap.shape)
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # # Display Grad CAM
    # display(Image(cam_path))
    #return superimposed_img

def predict(audio_path,number):
    spectrogram, _ = preprocess(audio_path, number)
    xval = tf.expand_dims(spectrogram, axis=0)
    model = tf.keras.models.load_model('my_model2')
    # model.summary()

    yhat = model.predict(xval)

    if yhat[0][0] > 0.99:
        print("Predition",1)
        prediction = 1
    else:
        print("Predition",0)
        prediction = 0

    print("xval ",xval.shape)
    last_conv_layer_name = "conv2d_2"

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    # preds = model.predict(img_array)
    # print("Predicted:", preds)

    # Generate class activation heatmap 
    heatmap = make_gradcam_heatmap(xval, model, last_conv_layer_name)

    save_and_display_gradcam(tf.transpose(xval,perm=[1,2,0]),heatmap,alpha = 0.4)
    return prediction


