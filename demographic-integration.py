import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers

# Load demographic data from CSV
# Example format:
# user_id, age, gender, location, previous_interactions, preferred_style
# 1, 25, 'male', 'urban', [previous design ratings encoded], 'minimalist'
df = pd.read_csv('demographic_data.csv')

# Process demographic data
def process_demographic_data(df):
    # Separate features and target
    X = df.drop(['user_id', 'preferred_style'], axis=1)
    y = df['preferred_style']
    
    # Create preprocessing pipeline for demographic data
    numeric_features = ['age'] + [col for col in X.columns if 'rating' in col or 'interaction' in col]
    categorical_features = ['gender', 'location']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # One-hot encode the target variable
    style_encoder = OneHotEncoder(sparse=False)
    y_train_encoded = style_encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_val_encoded = style_encoder.transform(y_val.values.reshape(-1, 1))
    y_test_encoded = style_encoder.transform(y_test.values.reshape(-1, 1))
    
    return (X_train_processed, y_train_encoded, 
            X_val_processed, y_val_encoded, 
            X_test_processed, y_test_encoded, 
            preprocessor, style_encoder)

# Create a demographic-based model
def create_demographic_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Combine image-based and demographic models
def create_combined_model(img_model, demo_model, num_classes):
    # Get the penultimate layer from the image model
    img_features = img_model.layers[-2].output
    
    # Input layer for demographic data
    demo_input = layers.Input(shape=(demo_model.input_shape[1],))
    demo_features = demo_model.layers[0](demo_input)
    demo_features = demo_model.layers[1](demo_features)
    demo_features = demo_model.layers[2](demo_features)
    demo_features = demo_model.layers[3](demo_features)
    
    # Concatenate both feature sets
    combined = layers.Concatenate()([img_features, demo_features])
    
    # Add additional layers for combined processing
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the combined model
    combined_model = models.Model(
        inputs=[img_model.input, demo_input],
        outputs=outputs
    )
    
    combined_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return combined_model

# Implementation for combined prediction workflow
def combined_predict(img_path, demographic_data, img_model, demo_preprocessor, style_encoder, combined_model):
    # Process the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Process demographic data
    demo_processed = demo_preprocessor.transform(pd.DataFrame([demographic_data]))
    
    # Use the combined model for prediction
    predictions = combined_model.predict([img_array, demo_processed])
    
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Convert the index back to class name
    class_names = style_encoder.categories_[0]
    predicted_style = class_names[predicted_class_idx]
    
    return {
        'style': predicted_style,
        'confidence': float(confidence),
        'all_probabilities': {style: float(prob) for style, prob in zip(class_names, predictions[0])}
    }

# Main workflow for training the combined model
def train_combined_model():
    # Load the pretrained image model
    img_model = tf.keras.models.load_model('design_preferences_model.h5')
    
    # Process demographic data
    (X_train, y_train, 
     X_val, y_val, 
     X_test, y_test, 
     preprocessor, style_encoder) = process_demographic_data(pd.read_csv('demographic_data.csv'))
    
    # Create and train demographic model
    num_classes = y_train.shape[1]
    demo_model = create_demographic_model(X_train.shape[1], num_classes)
    
    print("Training demographic model...")
    demo_history = demo_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Create combined model
    combined_model = create_combined_model(img_model, demo_model, num_classes)
    
    # Create custom data generator to feed both inputs
    def create_combined_generator(img_generator, X_demo, y_demo):
        demo_idx = 0
        for img_batch, y_img_batch in img_generator:
            current_batch_size = len(y_img_batch)
            # Take the corresponding number of demographic samples
            X_demo_batch = X_demo[demo_idx:demo_idx + current_batch_size]
            y_demo_batch = y_demo[demo_idx:demo_idx + current_batch_size]
            demo_idx = (demo_idx + current_batch_size) % len(X_demo)
            
            # Return combined inputs and labels
            yield [img_batch, X_demo_batch], y_demo_batch
    
    # Train the combined model
    print("Training combined model...")
    combined_history = combined_model.fit(
        create_combined_generator(train_ds, X_train, y_train),
        epochs=15,
        validation_data=create_combined_generator(val_ds, X_val, y_val),
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating combined model...")
    combined_test_generator = create_combined_generator(test_ds, X_test, y_test)
    test_loss, test_accuracy = combined_model.evaluate(
        combined_test_generator,
        steps=len(test_ds),
        verbose=1
    )
    
    print(f"Combined model test accuracy: {test_accuracy:.4f}")
    
    # Save the models and preprocessors
    combined_model.save('combined_design_preferences_model.h5')
    demo_model.save('demographic_model.h5')
    
    # Save preprocessors (using pickle)
    import pickle
    with open('demographic_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    with open('style_encoder.pkl', 'wb') as f:
        pickle.dump(style_encoder, f)
    
    print("Combined model training complete!")
    
    return combined_model, preprocessor, style_encoder

if __name__ == "__main__":
    train_combined_model()
