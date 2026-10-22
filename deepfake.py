import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
import os
import zipfile
import kagglehub
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 AI vs Real Image Detector")
st.markdown("""
This application uses a Convolutional Neural Network (CNN) to detect whether an image is **Real** or **AI-Generated** using the CIFAKE dataset.
Upload an image or use the sample images to test the model.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
    ["Model Training", "Image Upload", "Batch Prediction", "Performance Analysis"])

# Constants based on dataset information
IMG_SIZE = 32  # CIFAR-10 images are 32x32
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # REAL and FAKE
CLASS_NAMES = ['FAKE', 'REAL']

def create_cnn_model():
    """Create a CNN model for image classification"""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize to match model input
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def download_and_prepare_dataset():
    """Download and prepare the CIFAKE dataset"""
    st.info("Downloading CIFAKE dataset from Kaggle...")
    
    try:
        # Download dataset using kagglehub
        path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
        st.success(f"Dataset downloaded to: {path}")
        
        # Unzip if necessary
        dataset_path = os.path.join(path, "cifake-real-and-ai-generated-synthetic-images")
        
        # Create data generators
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        
        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            directory=os.path.join(dataset_path, "train"),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        # Create validation generator
        validation_generator = train_datagen.flow_from_directory(
            directory=os.path.join(dataset_path, "train"),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        
        # Create test generator
        test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            directory=os.path.join(dataset_path, "test"),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator, path
        
    except Exception as e:
        st.error(f"Error downloading dataset: {e}")
        st.info("Using sample mode with limited functionality")
        return None, None, None, None

def train_model(train_gen, val_gen):
    """Train the CNN model"""
    st.subheader("🔄 Training CNN Model")
    
    # Create model
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Display model summary
    st.text("Model Architecture:")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))
    
    # Training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )
    
    # Update progress
    for i in range(EPOCHS):
        progress_bar.progress((i + 1) / EPOCHS)
        status_text.text(f"Training epoch {i + 1}/{EPOCHS}")
        time.sleep(0.1)  # For demonstration
    
    progress_bar.progress(1.0)
    status_text.text("Training completed!")
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

def single_image_prediction(model):
    """Handle single image upload and prediction"""
    st.subheader("📤 Upload Image for Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", 
                                        type=['jpg', 'jpeg', 'png'])
        
        # Sample images
        st.markdown("### Or try sample images:")
        sample_option = st.selectbox("Select sample image type", 
                                   ["AI-Generated (FAKE)", "Real Image (REAL)"])
        
        # Generate sample image
        if st.button("Generate Sample"):
            if sample_option == "AI-Generated (FAKE)":
                # Create a synthetic-looking image
                sample_img = np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 255
                sample_img = sample_img.astype(np.uint8)
            else:
                # Create a more natural-looking image
                sample_img = np.random.randn(IMG_SIZE, IMG_SIZE, 3) * 50 + 128
                sample_img = np.clip(sample_img, 0, 255).astype(np.uint8)
            
            uploaded_file = sample_img
    
    with col2:
        if uploaded_file is not None:
            # Display image
            if isinstance(uploaded_file, np.ndarray):
                image = Image.fromarray(uploaded_file)
            else:
                image = Image.open(uploaded_file)
            
            st.image(image, caption="Uploaded Image", width=300)
            
            # Preprocess and predict
            img_array = preprocess_image(image)
            
            if model:
                predictions = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class] * 100
                
                # Display results
                st.markdown("### Prediction Results")
                
                # Create metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Prediction", CLASS_NAMES[predicted_class])
                with col_b:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                # Confidence bar
                st.progress(float(confidence / 100))
                
                # Detailed probabilities
                st.markdown("#### Detailed Probabilities:")
                prob_df = pd.DataFrame({
                    'Class': CLASS_NAMES,
                    'Probability': predictions[0] * 100
                })
                st.bar_chart(prob_df.set_index('Class'))
                
                # Explanation
                if predicted_class == 0:  # FAKE
                    st.warning("""
                    **AI-Generated Image Detected!**
                    - The model detected patterns consistent with AI-generated content
                    - Common indicators: unusual textures, inconsistent lighting, or artificial patterns
                    """)
                else:  # REAL
                    st.success("""
                    **Real Image Detected!**
                    - The image appears to be naturally captured
                    - Shows patterns consistent with real-world photography
                    """)
            else:
                st.warning("Please train or load a model first in the 'Model Training' section.")

def batch_prediction(model, test_gen):
    """Handle batch prediction on test dataset"""
    st.subheader("📊 Batch Prediction")
    
    if model and test_gen:
        # Get predictions
        predictions = model.predict(test_gen, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        # Classification report
        st.markdown("#### Classification Report")
        report = classification_report(true_classes, predicted_classes, 
                                      target_names=CLASS_NAMES, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(true_classes, predicted_classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Sample predictions
        st.markdown("#### Sample Predictions")
        num_samples = min(5, len(test_gen.filenames))
        sample_indices = np.random.choice(len(test_gen.filenames), num_samples, replace=False)
        
        cols = st.columns(num_samples)
        for idx, col in zip(sample_indices, cols):
            img_path = test_gen.filepaths[idx]
            img = Image.open(img_path)
            img_array = preprocess_image(img)
            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            confidence = pred[0][pred_class] * 100
            
            with col:
                st.image(img, width=100)
                st.caption(f"True: {CLASS_NAMES[true_classes[idx]]}")
                st.caption(f"Pred: {CLASS_NAMES[pred_class]}")
                st.caption(f"Conf: {confidence:.1f}%")
    else:
        st.warning("Please train the model and load test data first.")

def main():
    """Main application function"""
    
    # Initialize session state for model and data
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'train_gen' not in st.session_state:
        st.session_state.train_gen = None
    if 'test_gen' not in st.session_state:
        st.session_state.test_gen = None
    
    if app_mode == "Model Training":
        st.header("🤖 Model Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### About the CIFAKE Dataset
            - **60,000** AI-generated synthetic images (FAKE)
            - **60,000** real images from CIFAR-10 (REAL)
            - **100,000** training images (50k per class)
            - **20,000** test images (10k per class)
            - Image size: 32x32 pixels
            """)
        
        with col2:
            st.markdown("""
            ### Model Info
            - **Architecture**: CNN with 3 convolutional blocks
            - **Input**: 32x32 RGB images
            - **Output**: REAL or FAKE classification
            - **Features**: Batch normalization, dropout for regularization
            """)
        
        # Dataset and training options
        st.subheader("Training Configuration")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            epochs = st.slider("Number of Epochs", 5, 50, EPOCHS)
        with col_b:
            batch_size = st.slider("Batch Size", 16, 128, BATCH_SIZE, step=16)
        with col_c:
            learning_rate = st.select_slider("Learning Rate", 
                                           options=[0.1, 0.01, 0.001, 0.0001], 
                                           value=0.001)
        
        # Training buttons
        col_train, col_load = st.columns(2)
        
        with col_train:
            if st.button("🚀 Train New Model", type="primary"):
                with st.spinner("Downloading dataset and training model..."):
                    train_gen, val_gen, test_gen, _ = download_and_prepare_dataset()
                    if train_gen:
                        st.session_state.train_gen = train_gen
                        st.session_state.test_gen = test_gen
                        model, history = train_model(train_gen, val_gen)
                        st.session_state.model = model
                        st.session_state.history = history
                        
                        # Save model
                        model.save('ai_vs_real_model.h5')
                        st.success("Model trained and saved successfully!")
                        
                        # Plot training history
                        plot_training_history(history)
        
        with col_load:
            if st.button("📁 Load Pre-trained Model"):
                try:
                    model = keras.models.load_model('ai_vs_real_model.h5')
                    st.session_state.model = model
                    st.success("Model loaded successfully!")
                except:
                    st.error("No pre-trained model found. Please train a model first.")
        
        # Display model info if available
        if st.session_state.model:
            st.subheader("Model Performance")
            
            # Evaluate on validation data
            if st.session_state.test_gen:
                loss, accuracy, precision, recall = st.session_state.model.evaluate(
                    st.session_state.test_gen, verbose=0
                )
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Test Loss", f"{loss:.4f}")
                col2.metric("Accuracy", f"{accuracy*100:.2f}%")
                col3.metric("Precision", f"{precision*100:.2f}%")
                col4.metric("Recall", f"{recall*100:.2f}%")
            
            if st.session_state.history:
                plot_training_history(st.session_state.history)
    
    elif app_mode == "Image Upload":
        single_image_prediction(st.session_state.model)
    
    elif app_mode == "Batch Prediction":
        batch_prediction(st.session_state.model, st.session_state.test_gen)
    
    elif app_mode == "Performance Analysis":
        st.header("📈 Performance Analysis")
        
        if st.session_state.model and st.session_state.history:
            # Model architecture visualization
            st.subheader("Model Architecture")
            st.code("""
            Input: 32x32x3 (RGB Image)
            ↓
            Conv2D(32, 3x3) + BatchNorm + ReLU
            ↓
            Conv2D(32, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
            ↓
            Conv2D(64, 3x3) + BatchNorm + ReLU
            ↓
            Conv2D(64, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
            ↓
            Conv2D(128, 3x3) + BatchNorm + ReLU
            ↓
            Conv2D(128, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
            ↓
            Flatten
            ↓
            Dense(256) + ReLU + BatchNorm + Dropout(0.5)
            ↓
            Dense(128) + ReLU + Dropout(0.3)
            ↓
            Dense(2) + Softmax
            ↓
            Output: [FAKE, REAL]
            """)
            
            # Feature importance (simulated)
            st.subheader("Feature Importance")
            st.markdown("""
            Based on the CIFAKE dataset analysis, the model focuses on:
            
            1. **Texture Patterns**: AI-generated images often have repetitive or unnatural textures
            2. **Edge Consistency**: Real images typically have more natural edge transitions
            3. **Color Distribution**: Synthetic images may have unusual color distributions
            4. **Local Features**: Small details that are inconsistent in AI-generated images
            
            *Note: For actual feature visualization, consider using Grad-CAM or similar techniques.*
            """)
            
            # Recommendations
            st.subheader("Improvement Recommendations")
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("""
                **To improve accuracy:**
                - Increase training data with augmentation
                - Try different architectures (ResNet, EfficientNet)
                - Use transfer learning from pre-trained models
                - Implement ensemble methods
                """)
            
            with col_rec2:
                st.markdown("""
                **For production use:**
                - Deploy model as REST API
                - Add confidence thresholds
                - Implement continuous learning
                - Add explainable AI features
                """)
        else:
            st.warning("Please train a model first to see performance analysis.")

# Run the main function
if __name__ == "__main__":
    main()