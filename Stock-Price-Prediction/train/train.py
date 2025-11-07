import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, x_train, y_train, x_val, y_val, save_path='models/best_model.h5'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    return history
