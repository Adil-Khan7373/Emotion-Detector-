from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Preprocessing the images
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    r'C:\Users\USER\Desktop\Projects\Emotion recognizer\Emotion_Recognizer\datasets\train 2',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    r'C:\Users\USER\Desktop\Projects\Emotion recognizer\Emotion_Recognizer\datasets\val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Building the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Assuming 5 classes for emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Saving the model
model.save('emotion_recognizer_model.h5')

# Evaluating the model
test_generator = datagen.flow_from_directory(
    r'C:\Users\USER\Desktop\Projects\Emotion recognizer\Emotion_Recognizer\datasets\test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
