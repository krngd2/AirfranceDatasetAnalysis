import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pyiqa
import torch

class VGGPerceptualMetric:
    def __init__(self, layer_names=None):
        """
        Initializes the perceptual metric.
        :param layer_names: A list of VGG16 layer names to use for feature extraction.
                            If None, uses default layers from block 1 to 5.
        """
        # Load VGG16 model, pre-trained on ImageNet
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        if layer_names is None:
            # Default layers, one from each convolutional block
            layer_names = [
                'block1_conv2', 'block2_conv2', 'block3_conv3',
                'block4_conv3', 'block5_conv3'
            ]

        # Create a new model that outputs the feature maps from the specified layers
        layer_outputs = [base_model.get_layer(name).output for name in layer_names]
        self.feature_extractor = Model(inputs=base_model.input, outputs=layer_outputs)

        # Initialize the GMSD metric from pyiqa to compare feature maps
        # We will use one instance per layer
        self.gmsd_metrics = [pyiqa.create_metric('gmsd', as_loss=True, device='cpu') for _ in layer_names]
        print(f"Initialized perceptual metric using VGG16 layers: {layer_names}")

    def _normalize_features(self, features):
        """Normalize feature maps along the channel dimension."""
        return features / (tf.sqrt(tf.reduce_sum(tf.square(features), axis=-1, keepdims=True)) + 1e-10)

    def calculate_distance(self, img1, img2):
        """
        Calculates the perceptual distance between two images.
        Images should be numpy arrays of shape (H, W, 3) with pixel values in [0, 255].
        """
        # Preprocess images for VGG16
        # Add a batch dimension and apply VGG preprocessing
        img1_processed = preprocess_input(np.expand_dims(img1, axis=0))
        img2_processed = preprocess_input(np.expand_dims(img2, axis=0))

        # Extract features for both images
        features1 = self.feature_extractor(img1_processed)
        features2 = self.feature_extractor(img2_processed)

        # Normalize features
        norm_features1 = [self._normalize_features(f) for f in features1]
        norm_features2 = [self._normalize_features(f) for f in features2]

        total_distance = 0.0
        layer_distances = []

        # Calculate GMSD for the feature maps of each layer and sum them up
        for i in range(len(self.gmsd_metrics)):
            # GMSD expects 3-channel RGB images, but feature maps have varying channels
            # We need to reduce the feature maps to 3 channels before applying GMSD
            
            # Permute from (B, H, W, C) to (B, C, H, W) for pyiqa if needed
            f1 = tf.transpose(norm_features1[i], perm=[0, 3, 1, 2])
            f2 = tf.transpose(norm_features2[i], perm=[0, 3, 1, 2])

            # Convert TensorFlow tensors to PyTorch tensors
            f1_torch = torch.from_numpy(f1.numpy())
            f2_torch = torch.from_numpy(f2.numpy())
            
            # Reduce to 3 channels by selecting the first 3 channels or using interpolation
            num_channels = f1_torch.shape[1]
            if num_channels >= 3:
                # Take the first 3 channels
                f1_rgb = f1_torch[:, :3, :, :]
                f2_rgb = f2_torch[:, :3, :, :]
            else:
                # If less than 3 channels, repeat channels to make it 3
                f1_rgb = f1_torch.repeat(1, 3 // num_channels + 1, 1, 1)[:, :3, :, :]
                f2_rgb = f2_torch.repeat(1, 3 // num_channels + 1, 1, 1)[:, :3, :, :]
            
            # Normalize to [0, 1] range as expected by GMSD
            # Use the same normalization parameters for both tensors to ensure identical inputs give zero distance
            combined = torch.cat([f1_rgb, f2_rgb], dim=0)
            global_min = combined.min()
            global_max = combined.max()
            
            f1_rgb = (f1_rgb - global_min) / (global_max - global_min + 1e-8)
            f2_rgb = (f2_rgb - global_min) / (global_max - global_min + 1e-8)

            # Use the forward method of the underlying model to bypass the image reading logic
            dist = self.gmsd_metrics[i].net(f1_rgb, f2_rgb)
            
            # dist is a torch.Tensor, get its scalar value
            dist_val = dist.item()
            layer_distances.append(dist_val)
            total_distance += dist_val

        return total_distance

# Example Usage:
if __name__ == '__main__':
    # Create two dummy images (e.g., random noise) for demonstration
    # In your case, you would load your actual CFD images here
    # Ensure images are resized to (224, 224)
    dummy_img1 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_img2 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_img3 = dummy_img1.copy() # Identical image

    # Initialize the metric
    perceptual_metric = VGGPerceptualMetric()

    # Calculate distance between different images
    distance1_2 = perceptual_metric.calculate_distance(dummy_img1, dummy_img2)
    print(f"Perceptual distance between img1 and img2: {distance1_2}")

    # Calculate distance between identical images (should be close to 0)
    distance1_3 = perceptual_metric.calculate_distance(dummy_img1, dummy_img3)
    print(f"Perceptual distance between img1 and img3 (identical): {distance1_3}")