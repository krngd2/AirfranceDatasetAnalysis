import tensorflow as tf
import torch
import numpy as np
import argparse

def convert_keras_to_pytorch(keras_model_path, pytorch_model_path):
    """
    Loads weights from a Keras VGG16 .h5 file, converts them to a PyTorch state_dict,
    and saves them to a .pth file.
    """
    try:
        keras_model = tf.keras.models.load_model(keras_model_path)
        print("Keras model loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    pytorch_state_dict = {}
    
    # Mapping from Keras layer names to PyTorch layer names in lpips.vgg16
    # This needs to match the structure of the vgg16 model used by lpips
    # The lpips vgg16 features are named like 'slice1.0', 'slice1.2', etc.
    # Let's create a dummy lpips vgg to inspect layer names
    from lpips.pretrained_networks import vgg16 as lpips_vgg
    pytorch_vgg = lpips_vgg(pretrained=False)
    
    # The vgg16 model in lpips is a ModuleList of Sequential layers (slices)
    # We need to iterate through the modules to find the Conv2d layers
    pytorch_conv_layers = []
    for module in pytorch_vgg.modules():
        if isinstance(module, torch.nn.Conv2d):
            pytorch_conv_layers.append(module)
    
    keras_conv_layer_names = sorted([layer.name for layer in keras_model.layers if 'conv' in layer.name and 'block' in layer.name])

    if len(keras_conv_layer_names) != len(pytorch_conv_layers):
        print(f"Warning: Number of Keras conv layers ({len(keras_conv_layer_names)}) does not match PyTorch conv layers ({len(pytorch_conv_layers)}).")
        return

    # This mapping is based on the standard VGG16 architecture and how lpips names its layers
    # It might need adjustment if your model or lpips version is different.
    layer_name_map = {}
    pytorch_conv_idx = 0
    # We need to find the names of the layers in the state_dict
    # The names are based on the module structure, e.g., '0.0.weight' for the first conv in the first slice
    for i, module in enumerate(pytorch_vgg.children()): # iterate through slices
        for j, layer in enumerate(module.children()): # iterate through layers in slice
            if isinstance(layer, torch.nn.Conv2d):
                if pytorch_conv_idx < len(keras_conv_layer_names):
                    pytorch_layer_name = f'{i}.{j}'
                    keras_layer_name = keras_conv_layer_names[pytorch_conv_idx]
                    layer_name_map[keras_layer_name] = pytorch_layer_name
                    pytorch_conv_idx += 1

    for keras_layer_name, pytorch_layer_base_name in layer_name_map.items():
        keras_layer = keras_model.get_layer(keras_layer_name)
        weights, biases = keras_layer.get_weights()
        
        # Convert Keras weights to PyTorch format
        weights_tensor = torch.from_numpy(weights).permute(3, 2, 0, 1)
        biases_tensor = torch.from_numpy(biases)
        
        pytorch_state_dict[f'{pytorch_layer_base_name}.weight'] = weights_tensor
        pytorch_state_dict[f'{pytorch_layer_base_name}.bias'] = biases_tensor
        print(f"Converted weights for {keras_layer_name} -> {pytorch_layer_base_name}")

    try:
        torch.save(pytorch_state_dict, pytorch_model_path)
        print(f"PyTorch model weights saved to {pytorch_model_path}")
    except Exception as e:
        print(f"Error saving PyTorch model: {e}")

if __name__ == '__main__':
    # Path to your Keras .h5 model
    keras_model_path = './vgg16_finetuned_comparitive.h5'
    # Path to save the converted PyTorch weights
    parser = argparse.ArgumentParser(description='Convert Keras VGG weights to PyTorch LPIPS format.')
    parser.add_argument('keras_model_path', help='Path to the input Keras .h5 model file.')
    parser.add_argument('pytorch_weights_path', help='Path for the output PyTorch .pth weights file.')
    args = parser.parse_args()

    keras_model_path = args.keras_model_path if args.keras_model_path else keras_model_path
    pytorch_weights_path = args.pytorch_weights_path if args.pytorch_weights_path else './vgg16_finetuned_comparitive.pth'
    
    convert_keras_to_pytorch(keras_model_path, pytorch_weights_path)
