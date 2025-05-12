import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import io

def generate_gradcam_map(model, image_tensor, class_idx, image_size=224):
    model.eval()

    # Ensure image_tensor has the correct shape
    if len(image_tensor.shape) == 3:  # Shape: (channels, height, width)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: (1, channels, height, width)
    elif len(image_tensor.shape) != 4:  # Shape must be (batch_size, channels, height, width)
        raise ValueError(f"Expected image_tensor to have 4 dimensions (batch_size, channels, height, width), got shape {image_tensor.shape}")

    image_tensor.requires_grad_(True)

    # Forward pass to get logits
    outputs = model(image_tensor)

    # Backward pass for the specific class
    model.zero_grad()
    target = outputs[0, class_idx]
    target.backward()

    # Access the ViT's last layer attention weights
    with torch.no_grad():
        # model.vit is a ViTModel, so we access embeddings and encoder directly
        embeddings = model.vit.embeddings(image_tensor)  # Shape: (batch_size, seq_len, hidden_size)
        # Forward pass with output_attentions=True to get attention weights
        vit_outputs = model.vit(image_tensor, output_attentions=True)
        attention_weights = vit_outputs.attentions[-1]  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = attention_weights.mean(dim=1)  # Average over heads: (batch_size, seq_len, seq_len)
        cls_attention = attention_weights[0, 0, 1:]  # CLS token's attention to patches

    # Reshape attention to match patch grid
    patch_size = 16  # 224/16 = 14 patches per dimension
    num_patches = (image_size // patch_size) ** 2
    attention_map = cls_attention[:num_patches].view(image_size // patch_size, image_size // patch_size)

    # Upsample to image size
    attention_map_resized = attention_map.cpu().numpy()
    attention_map_resized = np.array(Image.fromarray(attention_map_resized).resize((image_size, image_size), Image.BILINEAR))

    # Normalize
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min() + 1e-8)

    # Convert image tensor to numpy for overlay
    image = image_tensor.squeeze(0).detach().cpu()
    image = image.permute(1, 2, 0).numpy()
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    image = image.astype(np.uint8)

    # Create heatmap with adjusted alpha for better clarity
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.7)  # Increased alpha for more visibility
    plt.axis('off')
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    gradcam_image = Image.open(buf)

    # Return both the Grad-CAM image and the attention map
    return gradcam_image, attention_map_resized