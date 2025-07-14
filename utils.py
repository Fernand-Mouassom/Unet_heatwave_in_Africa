from tensorflow.keras import layers, models, Input, regularizers, metrics
from tensorflow.keras.optimizers import Adam

def zscore(da):
    return (da - da.mean("time")) / da.std("time")

def with_channel(da, name):
    return da.expand_dims("channel", axis=-1).assign_coords(channel=[name])

def unet_regressor(input_shape=(321, 321, 9), n_filters=32, dropout_rate=0.3, l2_strength=1e-4, lr=1e-3):
    reg = regularizers.l2(l2_strength)
    inputs = Input(shape=input_shape)

    # Encoder block 1
    c1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout_rate)(p1)

    # Encoder block 2
    c2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout_rate)(p2)

    # Encoder block 3
    c3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(dropout_rate)(p3)

    # Bottleneck
    b = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(p3)
    b = layers.BatchNormalization()(b)
    b = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(b)
    b = layers.BatchNormalization()(b)

    # Decoder block 1
    u3 = layers.UpSampling2D((2, 2))(b)
    u3 = layers.Concatenate()([u3, c3])
    u3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(u3)
    u3 = layers.BatchNormalization()(u3)
    u3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(u3)
    u3 = layers.BatchNormalization()(u3)
    u3 = layers.Dropout(dropout_rate)(u3)

    # Decoder block 2
    u2 = layers.UpSampling2D((2, 2))(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(u2)
    u2 = layers.BatchNormalization()(u2)
    u2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(u2)
    u2 = layers.BatchNormalization()(u2)
    u2 = layers.Dropout(dropout_rate)(u2)

    # Decoder block 3
    u1 = layers.UpSampling2D((2, 2))(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(u1)
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same', activity_regularizer=reg)(u1)
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.Dropout(dropout_rate)(u1)

    # Output layer with sigmoid activation for binary classification
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(u1)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=METRICS
    )
    return model
