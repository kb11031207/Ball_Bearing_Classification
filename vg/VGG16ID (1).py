import os
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, models, layers
import sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def make_generator():
    # Simple scaling and default batchsize.  Don't augment nontraining data
    noaug_datagen = ImageDataGenerator(rescale=1./255)  

    # Use data_simplified/test directory by default if no argument is provided
    test_dir = sys.argv[2] if len(sys.argv) > 2 else 'data_simplified2/test'
    
    # Change to categorical for multi-class
    test_generator = noaug_datagen.flow_from_directory(
        test_dir,              
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        shuffle=False)  # Don't shuffle to maintain correspondence with filenames

    return test_generator
    
def evaluate_all_test_data(model, test_generator, class_names):
    # Reset the generator to the beginning
    test_generator.reset()
    
    # Get the total number of samples
    num_samples = test_generator.samples
    steps = np.ceil(num_samples / test_generator.batch_size).astype(int)
    
    # Get predictions for all test data
    all_predictions = []
    all_true_labels = []
    file_paths = []
    
    for i in range(steps):
        batch_x, batch_y = next(test_generator)
        batch_pred = model.predict(batch_x)
        batch_pred_classes = np.argmax(batch_pred, axis=1)
        
        # Store predictions and true labels
        all_predictions.extend(batch_pred_classes)
        all_true_labels.extend(np.argmax(batch_y, axis=1))
        
        # Get current batch filenames
        batch_indices = range(
            i * test_generator.batch_size,
            min((i + 1) * test_generator.batch_size, num_samples)
        )
        file_paths.extend([test_generator.filenames[j] for j in batch_indices])
    
    # Trim to actual number of samples
    all_predictions = all_predictions[:num_samples]
    all_true_labels = all_true_labels[:num_samples]
    file_paths = file_paths[:num_samples]
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Display classification report
    report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=class_names
    )
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Show examples of misclassified images
    print("\nShowing examples of misclassifications...")
    misclassified_indices = np.where(np.array(all_predictions) != np.array(all_true_labels))[0]
    
    if len(misclassified_indices) > 0:
        # Show up to 5 misclassified images
        num_to_show = min(5, len(misclassified_indices))
        for i in range(num_to_show):
            idx = misclassified_indices[i]
            img_path = file_paths[idx]
            true_class = class_names[all_true_labels[idx]]
            pred_class = class_names[all_predictions[idx]]
            
            # Load and display the image
            img = plt.imread(os.path.join(test_generator.directory, img_path))
            plt.figure()
            plt.imshow(img)
            plt.title(f"True: {true_class}, Predicted: {pred_class}\nPath: {img_path}")
            plt.axis('off')
            plt.show()
    else:
        print("No misclassifications found!")
    
    return accuracy, all_predictions, all_true_labels

def show_sample_predictions(model, test_generator, class_names, num_samples=10):
    test_generator.reset()
    batch_x, batch_y = next(test_generator)
    
    # Make predictions
    predictions = model.predict(batch_x)
    
    # Display sample predictions (up to num_samples)
    samples_to_show = min(num_samples, len(batch_x))
    for i in range(samples_to_show):
        plt.figure()
        plt.imshow(batch_x[i])
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[i])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[i][predicted_class_idx]
        
        # Get true class
        true_class_idx = np.argmax(batch_y[i])
        true_class = class_names[true_class_idx]
        
        # Set title color based on correctness
        color = 'green' if predicted_class_idx == true_class_idx else 'red'
        
        plt.title(f"True: {true_class}\nPred: {predicted_class} ({confidence:.4f})", 
                 color=color)
        plt.axis('off')
        plt.show()

def main():
    # Use default model path if not provided
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'material_model.h5'
    model = models.load_model(model_path)

    test_generator = make_generator()
    
    # Get class names from the test generator
    class_indices = test_generator.class_indices
    class_names = list(class_indices.keys())
    
    print(f"Model: {model_path}")
    print(f"Test directory: {test_generator.directory}")
    print(f"Classes: {class_names}")
    print(f"Total test samples: {test_generator.samples}")
    
    # First show some sample predictions
    print("\n=== Sample Predictions ===")
    show_sample_predictions(model, test_generator, class_names)
    
    # Then evaluate on the entire test set
    print("\n=== Full Evaluation ===")
    accuracy, predictions, true_labels = evaluate_all_test_data(model, test_generator, class_names)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python VGG16ID.py [model_path] [test_data_dir]")
        print("Using default paths: material_model.h5, data_simplified2/test")
    
    main()