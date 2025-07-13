import torch
import lpips
from lpips.pretrained_networks import vgg16
from lpips.lpips import spatial_average
from lpips import normalize_tensor

class CustomFeatureMetric(lpips.LPIPS):
    """
    Calculates a perceptual metric using a fine-tuned VGG16 backbone.

    This class computes the mean squared L2 distance between normalized
    feature maps directly, without using the pre-trained LPIPS calibration
    layers, which are invalid for a custom feature extractor.
    """
    def __init__(self, custom_vgg_path, net='vgg'):
        # Step 1: Initialize the parent class WITHOUT loading its linear layers.
        # We only use its structure and helper methods (like scaling_layer).
        super().__init__(net=net, lpips=False)

        # Step 2: Load and set your custom VGG backbone.
        custom_net = vgg16(pretrained=False, requires_grad=False)
        state_dict = torch.load(custom_vgg_path, map_location=torch.device('cpu'))
        custom_net.load_state_dict(state_dict, strict=False)
        self.net = custom_net

        self.eval()

    def forward(self, img1, img2, normalize=True):
        """
        Computes the distance. The `normalize` flag should be True if your
        input images are in the [0, 1] range.
        """
        # --- Normalize input images ---
        if normalize:
            img1 = self.scaling_layer(img1)
            img2 = self.scaling_layer(img2)

        # --- Extract features from all layers ---
        feats1 = self.net.forward(img1)
        feats2 = self.net.forward(img2)

        # --- Compute distance for each layer ---
        layer_distances = []
        for feat1, feat2 in zip(feats1, feats2):
            # Normalize features across the channel dimension, like in the original LPIPS
            norm_feat1 = normalize_tensor(feat1)
            norm_feat2 = normalize_tensor(feat2)

            # Calculate squared L2 distance
            dist = (norm_feat1 - norm_feat2) ** 2

            # Average spatially and then over the batch
            layer_dist = spatial_average(dist, keepdim=False).mean()
            layer_distances.append(layer_dist)

        # Average the distances from all layers to get the final score
        # final_dist = torch.mean(torch.stack(layer_distances))
        final_dist = torch.sum(torch.stack(layer_distances))

        return final_dist

# --- Example Usage ---
if __name__ == '__main__':
    # You may need to adjust this path based on your file structure
    my_vgg_model_path = '../vgg16_finetuned_remeshed.pth'

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize your new custom metric
        custom_metric_fn = CustomFeatureMetric(custom_vgg_path=my_vgg_model_path).to(device)

        # Example with dummy images
        img0 = torch.zeros(1, 3, 64, 64, device=device)
        img1 = torch.ones(1, 3, 64, 64, device=device)

        distance = custom_metric_fn(img0, img1)
        print(f"Distance with custom feature metric: {distance.item():.6f}")

        # For comparison, the standard LPIPS
        loss_fn_standard = lpips.LPIPS(net='vgg').to(device)
        distance_standard = loss_fn_standard(img0, img1)
        print(f"LPIPS distance with standard VGG:  {distance_standard.item():.6f}")

    except FileNotFoundError:
        print(f"Error: Model file not found at '{my_vgg_model_path}'. Please check the path.")