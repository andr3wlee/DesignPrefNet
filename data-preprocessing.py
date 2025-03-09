import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

def create_dataset_structure():
    """
    Create the necessary directory structure for the design preferences dataset.
    """
    # Define the style categories
    styles = ['minimalist', 'colorful', 'retro', 'sleek', 'futuristic']
    
    # Create main directory
    os.makedirs('design_preferences_dataset', exist_ok=True)
    
    # Create train, validation, and test directories
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join('design_preferences_dataset', split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create subdirectories for each style
        for style in styles:
            style_dir = os.path.join(split_dir, style)
            os.makedirs(style_dir, exist_ok=True)
    
    print("Dataset directory structure created successfully.")

def process_design_images(source_dir, target_dir='design_preferences_dataset', test_size=0.2, val_size=0.2):
    """
    Process design images from the source directory and organize them into train/val/test splits.
    
    Args:
        source_dir (str): Directory containing the design images organized by style
        target_dir (str): Target directory for the processed dataset
        test_size (float): Fraction of data to use for testing
        val_size (float): Fraction of training data to use for validation
    """
    styles = os.listdir(source_dir)
    
    # Create a dataframe to track image metadata
    metadata = []
    
    for style in styles:
        style_dir = os.path.join(source_dir, style)
        if not os.path.isdir(style_dir):
            continue
            
        # Get all images for this style
        images = [f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Train/val/test split
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=42)
        
        # Copy images to their respective directories
        for img in train_imgs:
            src = os.path.join(style_dir, img)
            dst = os.path.join(target_dir, 'train', style, img)
            shutil.copy(src, dst)
            metadata.append({'filename': img, 'style': style, 'split': 'train'})
            
        for img in val_imgs:
            src = os.path.join(style_dir, img)
            dst = os.path.join(target_dir, 'validation', style, img)
            shutil.copy(src, dst)
            metadata.append({'filename': img, 'style': style, 'split': 'validation'})
            
        for img in test_imgs:
            src = os.path.join(style_dir, img)
            dst = os.path.join(target_dir, 'test', style, img)
            shutil.copy(src, dst)
            metadata.append({'filename': img, 'style': style, 'split': 'test'})
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(target_dir, 'metadata.csv'), index=False)
    
    # Print dataset statistics
    print("Dataset processing complete!")
    print(f"Total images: {len(metadata_df)}")
    print(metadata_df.groupby(['split', 'style']).size())

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for the model.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image as numpy array
    """
    # Load and resize image
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = img.convert('RGB')  # Ensure image has 3 channels
    
    # Convert to array and preprocess
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array

def generate_synthetic_demographic_data(metadata_df, output_file='demographic_data.csv', num_users=1000):
    """
    Generate synthetic demographic data for simulating user profiles.
    
    Args:
        metadata_df (DataFrame): Image metadata DataFrame
        output_file (str): Output CSV file path
        num_users (int): Number of user profiles to generate
    """
    # Define possible values for categorical features
    genders = ['male', 'female', 'non-binary', 'other']
    locations = ['urban', 'suburban', 'rural']
    style_preferences = metadata_df['style'].unique()
    
    # Generate synthetic user profiles
    np.random.seed(42)  # For reproducibility
    
    user_data = []
    for user_id in range(1, num_users + 1):
        # Generate demographic attributes
        age = np.random.randint(18, 65)
        gender = np.random.choice(genders)
        location = np.random.choice(locations)
        
        # Generate synthetic interaction history
        # Each user has ratings (1-5) for different styles
        interaction_history = {f'{style}_rating': np.random.randint(1, 6) for style in style_preferences}
        
        # Determine preferred style based on highest rating
        primary_preference = max(interaction_history.items(), key=lambda x: x[1])[0].split('_')[0]
        
        # Create user profile
        user_profile = {
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'location': location,
            'preferred_style': primary_preference
        }
        user_profile.update(interaction_history)
        
        user_data.append(user_profile)
    
    # Create DataFrame and save to CSV
    user_df = pd.DataFrame(user_data)
    user_df.to_csv(output_file, index=False)
    
    print(f"Generated {num_users} synthetic user profiles and saved to {output_file}")
    print(user_df.head())
    
    # Plot distribution of styles
    plt.figure(figsize=(10, 6))
    sns.countplot(data=user_df, x='preferred_style')
    plt.title('Distribution of User Style Preferences')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('style_distribution.png')
    
    # Plot age distribution by style preference
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=user_df, x='preferred_style', y='age')
    plt.title('Age Distribution by Style Preference')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('age_by_style.png')
    
    return user_df

def visualize_dataset(dataset_dir='design_preferences_dataset'):
    """
    Visualize samples from the dataset.
    
    Args:
        dataset_dir (str): Directory containing the processed dataset
    """
    styles = os.listdir(os.path.join(dataset_dir, 'train'))
    
    plt.figure(figsize=(15, 10))
    for i, style in enumerate(styles):
        style_dir = os.path.join(dataset_dir, 'train', style)
        images = [f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        # Display 3 random images for each style
        for j in range(min(3, len(images))):
            img_path = os.path.join(style_dir, np.random.choice(images))
            img = Image.open(img_path)
            
            plt.subplot(len(styles), 3, i*3 + j + 1)
            plt.imshow(img)
            plt.title(f'{style}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.show()

if __name__ == "__main__":
    # Create the dataset directory structure
    create_dataset_structure()
    
    # Example usage - process images from a source directory
    # process_design_images('raw_design_images')
    
    # Load metadata after processing
    try:
        metadata_df = pd.read_csv('design_preferences_dataset/metadata.csv')
        # Generate synthetic demographic data
        generate_synthetic_demographic_data(metadata_df)
        # Visualize dataset samples
        visualize_dataset()
    except FileNotFoundError:
        print("Run the image processing step first to generate metadata.")
