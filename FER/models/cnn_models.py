from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


def create_emotion_classifier(input_shape=(224, 224, 3), num_classes=7):
    model = Sequential([
        # 1st Layer
        Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # 2nd Layer
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # 3rd Layer
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # 4th Layer
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Flatten Layer
        Flatten(),

        # Fully connected layer 1
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Fully connected layer 2
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compiling the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = create_emotion_classifier()
    model.summary()
