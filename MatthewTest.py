# MatthewTest.py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image




img_width, img_height = 200, 200

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 300
nb_validation_samples = 30
epochs = 10
batch_size = 10

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale = 1. /255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')



model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.summary()

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples)


model.save_weights('first_try.hS')
















img_pred = image.load_img('data/test/3_1_1_20170109191427473.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)

print (rslt[0][0], "[0]")


img_pred = image.load_img('data/test/3_1_1_20170109194527475.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[0]")


img_pred = image.load_img('data/test/3_1_2_20161219141805465.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[0]")


img_pred = image.load_img('data/test/3_1_2_20161219142051272.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[0]")

img_pred = image.load_img('data/test/15_0_0_20170104011743800.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")


img_pred = image.load_img('data/test/15_0_0_20170104012102240.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")


img_pred = image.load_img('data/test/15_0_0_20170104012346994.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")




img_pred = image.load_img('data/test/15_0_0_20170104012550546.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")


img_pred = image.load_img('data/test/29_1_0_20170103183824867.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")


img_pred = image.load_img('data/test/29_1_0_20170104021759835.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")


img_pred = image.load_img('data/test/29_1_0_20170105002624350.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")



img_pred = image.load_img('data/test/29_1_0_20170105163239939.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[1]")



img_pred = image.load_img('data/test/45_0_1_20170111200809203.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")




img_pred = image.load_img('data/test/45_0_2_20170104174321891.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")



img_pred = image.load_img('data/test/45_0_2_20170105173303117.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")


img_pred = image.load_img('data/test/45_0_2_20170109013208873.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")



img_pred = image.load_img('data/test/60_1_0_20170110154144201.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")




img_pred = image.load_img('data/test/60_1_0_20170110154145935.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")



img_pred = image.load_img('data/test/60_1_0_20170110154325940.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")



img_pred = image.load_img('data/test/60_1_0_20170110154613614.jpg.chip.jpg', target_size = (200,200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print (rslt[0][0], "[2]")