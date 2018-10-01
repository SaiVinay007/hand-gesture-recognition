import tensorflow as tf
import keras
import numpy as np
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.activations import relu ,softmax
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
from input_utils import flow_from_directory

from skimage import transform
from skimage.color import rgb2gray

#################### Loading images and the corresponding labels
def load_data(data_directory):

    images_directories = [d for d in os.listdir(data_directory)]

    images = []
    labels = []
    
    for i,d in enumerate(images_directories):
        path_of_d = os.path.join(data_directory,d) 
        files_paths = [os.path.join(path_of_d,f)
                        for f in os.listdir(path_of_d)
                        if f.endswith(".jpg")]
        
        for f in files_paths:
            images.append(cv2.imread(f))
            labels.append(d)

    return images,labels

root_path = '/home/saivinay/Documents/hand_gesture/data/'
train_data_directory = os.path.join(root_path,'train')
test_data_directory = os.path.join(root_path,'test')

images,labels = load_data(train_data_directory)

l=0
for i,label in enumerate(labels):
    if label=='A':
        l=0
    elif label=='B':
        l=1
    elif label=='C':
        l=2
    elif label=='Five':
        l=3
    elif label=='Point':
        l=4
    elif label=='V':
        l=5
    labels[i] = l      
# print(labels[655])

# labels = [l for i in labels
#             if i == "A"
#              l = 1]

# print(len(images))
# cv2.imshow('first_image',images[0])
# k = cv2.waitKey(10000) & 0xFF
# cv2.destroyAllWindows()


################# Defining a model #################
def Hand_gesture(input_tensor):
    base_model = ResNet50(input_tensor = input_tensor, include_top=False,weights=None)

    model = Model(inputs = base_model.input,outputs = base_model.output)
    for i in model.layers:
        i.trainable = False
    # model = Flatten()(model.output)
    model = Dense(units = 32,activation ='relu')(model.output)
    model = Dense(units =6,activation ='softmax')(model)

    model = Model(inputs = base_model.input,outputs = model)


    return (model)



'''
if __name__ == '__main__':

    # next_element, init_op = flow_from_directory('/home/saivinay/Documents/hand_gesture/train')
    # init = tf.global_variables_initializer()
    # sess =tf.Session()
    # sess.run(init)
    # sess.run(init_op)
    # print(sess.run(next_element[0][0][0]))
    # # print(sess.run(init_op))

    model = Hand_gesture()

    model.compile(optimizer ='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    
    batch_size = 16
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            '/home/saivinay/Documents/hand_gesture/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode=None)  # since we use binary_crossentropy loss, we need binary labels
    
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            '/home/saivinay/Documents/hand_gesture/test',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode=None)


    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    model.save_weights('first_try.h5')
'''

image28 = [transform.resize(image,(28,28)) for image in images]
image28 = np.array(image28)
# image28 = rgb2gray(image28)



epochs = 20
classes = 6
batch_size =32
learning_rate = 0.001


x = tf.placeholder(dtype = tf.float32,shape=[None, 28,28, 3]) 
y = tf.placeholder(dtype=tf.int32,shape=[6])
model = Hand_gesture(x)

logits = tf.reshape(model.output,shape=(-1,6))
tf.cast(logits,tf.int32)
# logits = model.output
# print(model.output.shape)
# print(y)
labels = keras.utils.to_categorical(labels, num_classes=6)
# print (y.get_shape())
# print(logits.get_shape())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = logits))

optimizer  = tf.train.AdamOptimizer(0.0001)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# print (model.summary())

for i in range(epochs):

    for X,Y in zip(image28,labels):
        
        _, loss_val  = sess.run([train_op, loss], feed_dict={x: X[None, :,:,:], y: Y})
        print("loss : ",loss_val)



sess.close()
    

    

