import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.cnn_models import *
from Data_preprocessing.data_preprocessing import *
from tensorflow.keras.models import Sequential

def load_existing_model(model_path):
    """
    Tải một mô hình đã tồn tại từ tệp .h5.

    Parameters:
    - model_path: Đường dẫn đến tệp .h5 chứa mô hình đã tồn tại.

    Returns:
    Mô hình đã tồn tại.
    """
    return load_model(model_path)


def train_model(train_dir, validation_dir, epochs=50, batch_size=32, model_save_path='models/emotion_classifier.keras'):
    """
    Huấn luyện mô hình phân loại cảm xúc.

    Parameters:
    - train_dir: Đường dẫn đến thư mục chứa dữ liệu huấn luyện.
    - validation_dir: Đường dẫn đến thư mục chứa dữ liệu validation.
    - epochs: Số epoch cho quá trình huấn luyện.
    - batch_size: Kích thước batch.
    - model_save_path: Đường dẫn để lưu mô hình đã huấn luyện.


    Returns:
    Lịch sử huấn luyện.
    """
    # Chuẩn bị dữ liệu
    train_generator, validation_generator = prepare_data(train_dir, validation_dir, batch_size)

    # Kiểm tra nếu đã cung cấp đường dẫn mô hình đã tồn tại
    if model_save_path:
        if os.path.exists(model_save_path):
            model = load_existing_model(model_save_path)
            print("Đã tải mô hình đã tồn tại từ:", model_save_path)
        else:
            print("Không tìm thấy mô hình đã tồn tại. Tạo một mô hình mới.")
            model = create_emotion_classifier(input_shape=(224, 224, 3), num_classes=7)
    else:
        print("Không có đường dẫn mô hình đã tồn tại được cung cấp. Tạo một mô hình mới.")
        model = create_emotion_classifier(input_shape=(224, 224, 3), num_classes=7)

    # Xác định các callbacks
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, early_stopping]
    )

    # In kết quả độ chính xác của quá trình huấn luyện
    training_accuracy = history.history['accuracy'][-1]
    validation_accuracy = history.history['val_accuracy'][-1]
    print(
        f"Quá trình huấn luyện đã hoàn tất. Độ chính xác trên tập huấn luyện: {training_accuracy:.2f}, trên tập validation: {validation_accuracy:.2f}")

    return history


if __name__ == '__main__':
    # Đường dẫn đến dữ liệu
    train_dir = 'data/images/train'
    validation_dir = 'data/images/validation'

    # Bắt đầu quá trình huấn luyện
    history = train_model(train_dir, validation_dir, epochs=100, model_save_path='models/emotion_classifier.keras')
    print("Quá trình huấn luyện đã hoàn tất.")
