import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle

print(f"tensorflow version {tf.__version__}")
print(f"Eager Execution Enabled: {tf.executing_eagerly()}\n")

devices = tf.config.get_visible_devices()
print(f"All Devices: \n{devices}\n")
print(f"Available GPUs: \n{tf.config.list_logical_devices('GPU')}\n")

# Load tf datasets
ds_directory = "../missouri_data_processed/tf_datasets/"
img_directory = "../missouri_data_processed/md_cropped/Set1"
batch_size = 32
image_size = 224
seed = 123
validation_split = 0.15

trainval_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
   img_directory,
   labels='inferred',
   label_mode='categorical',
   color_mode='rgb',
   batch_size=batch_size,
   image_size=(image_size, image_size),
   shuffle=True,
   seed=seed,
   validation_split=validation_split,
   subset="both",
   crop_to_aspect_ratio=False,
   )
class_names = trainval_ds.class_names

train_batches = tf.data.experimental.cardinality(trainval_ds)
val_ds = trainval_ds.take(train_batches // 5)
train_ds = trainval_ds.skip(train_batches // 5)

print('Number of train batches: %d' % tf.data.experimental.cardinality(train_ds))
print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

## Dataset loading not working so create in single file instead
# trainval_ds = tf.data.Dataset.load(ds_directory+"trainval_ds")
# test_ds = tf.data.Dataset.load(ds_directory+"test_ds")
# with open("../missouri_data_processed/class_names.pkl", 'rb') as f:
#    class_names = pickle.load(f)

# Make smaller datasets for testing
train_ds = train_ds.take(100)
val_ds = val_ds.take(25)
test_ds = test_ds.take(20)

# esnure images are correct size for model
train_ds = train_ds.map(lambda image, label: (tf.image.resize(image, (image_size, image_size)), label))
val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, (image_size, image_size)), label))
test_ds = test_ds.map(lambda image, label: (tf.image.resize(image, (image_size, image_size)), label))

# Plot training images as a sense check
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
   for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[tf.math.argmax(labels[i]).numpy()])
      plt.axis("off")
plt.savefig("train_examples.png")

# Optimize performance of tf datasets
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

img_rescaling = Sequential(
   [
      layers.Rescaling(1./255)
   ],
   name="img_rescaling",
)

def build_model(num_classes):
   inputs = layers.Input(shape=(image_size, image_size, 3))
   x = img_rescaling(inputs)
   model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

   # Freeze the pretrained weights
   model.trainable = False

   # Rebuild top
   x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
   x = layers.BatchNormalization()(x)

   top_dropout_rate = 0.2
   x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
   outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

   # Compile
   model = tf.keras.Model(inputs, outputs, name="EfficientNet")
   optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
   model.compile(
       optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
   )
   return model

def plot_hist(hist):
   plt.plot(hist.history["accuracy"])
   plt.plot(hist.history["val_accuracy"])
   plt.title("model accuracy")
   plt.ylabel("accuracy")
   plt.xlabel("epoch")
   plt.legend(["train", "validation"], loc="upper left")
   plt.savefig("training_history.png")


model = build_model(num_classes=len(class_names))
epochs = 25  # @param {type: "slider", min:8, max:80}
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
   "./tmp/checkpoint",
    save_best_only = True,
    save_weights_only = True
)
es_callback = tf.keras.callbacks.EarlyStopping(
    patience=4,
    restore_best_weights=True,
)
hist = model.fit(train_ds, 
                 epochs=epochs, 
                 validation_data=val_ds, 
                 verbose=2, 
                 callbacks=[model_checkpoint_callback, es_callback])
plot_hist(hist)

