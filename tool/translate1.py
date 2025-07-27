import math
from torchvision import transforms
from skimage import color, filters, morphology
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Keep ImageDraw, ImageFont for the example usage
import matplotlib.pyplot as plt
import torch


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


class SkimageSkeletonize:
    def __call__(self, img_pil):
        img = np.array(img_pil)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.ndim == 3 and img.shape[2] == 3:
            img = color.rgb2gray(img)
        elif img.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported image dimension/channels: {img.shape}")

        thresh = filters.threshold_otsu(img)
        binary = img < thresh

        skeleton = morphology.skeletonize(binary)

        skeleton_img = (skeleton * 255).astype(np.uint8)
        return Image.fromarray(skeleton_img)


def process_image(image, image_height, image_min_width, image_max_width):
    # Ensure input is a PIL Image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    elif isinstance(image, str): # Handle file path input
        image_pil = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("Input 'image' must be a PIL Image, NumPy array, or file path.")
    else:
        image_pil = image # It's already a PIL Image

    # Convert to RGB early if the original was not, ensuring consistent input for skeletonizer
    image_pil = image_pil.convert("RGB")
    
    # Apply Skeletonization
    skeletonizer = SkimageSkeletonize()
    skeletonized_image_pil = skeletonizer(image_pil)

    # Convert back to RGB after skeletonization, as `skeletonize` yields grayscale.
    # This prepares it for models expecting 3 channels.
    img_for_processing = skeletonized_image_pil.convert("RGB")

    # Apply your custom resize logic
    w, h = img_for_processing.size
    new_w, target_h = resize(w, h, image_height, image_min_width, image_max_width)

    # Perform the actual PIL Image resize
    img_for_processing = img_for_processing.resize((new_w, target_h), Image.LANCZOS)

    # Convert to NumPy array, transpose to CHW, and normalize to 0-1
    img_array = np.asarray(img_for_processing).transpose(2, 0, 1) # HWC to CHW
    img_array = img_array / 255.0 # Normalize

    # Convert to PyTorch tensor (common for deep learning models)
    img_tensor = torch.from_numpy(img_array).float()

    # Handle padding to `image_max_width` if the resized image is smaller
    current_w = img_tensor.shape[2] # Current width (C, H, W)
    if current_w < image_max_width:
        # Pad on the right side with zeros
        padding = (0, image_max_width - current_w)
        img_tensor = torch.nn.functional.pad(img_tensor, padding, "constant", 0)
    elif current_w > image_max_width:
        # If by some chance it's larger (e.g., due to rounding), crop it
        img_tensor = img_tensor[:, :, :image_max_width]

    return img_tensor


if __name__ == "__main__":
    dummy_image_path = "sample/20160722_0202_26749_1_tg_7.png" # Make sure this path exists or create a dummy image

    try:
        original_image = Image.open(dummy_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Không tìm thấy file '{dummy_image_path}'. Tạo ảnh trắng giả lập có chữ.")
        original_image = Image.new('RGB', (508, 1104), color='white')
        draw = ImageDraw.Draw(original_image)
        try:
            # Try loading a common system font, or provide a full path if necessary
            font_path = "arial.ttf" # Example for Windows, or use a path like "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" on Linux
            font = ImageFont.truetype(font_path, 40)
        except IOError:
            print(f"Không tìm thấy font '{font_path}'. Sử dụng font mặc định.")
            font = ImageFont.load_default()
        draw.text((10, 20), "Hello OCR!", fill=(0,0,0), font=font)

    # Define image processing parameters
    target_height = 508
    min_width = 1104
    max_width = 1104

    # Process the image using your function
    final_processed_tensor = process_image(original_image, target_height, min_width, max_width)

    print(f"\nKích thước ảnh gốc (PIL): {original_image.size}")
    print(f"Kích thước tensor sau xử lý (C, H, W): {final_processed_tensor.shape}")

    # --- Visualization ---
    # Convert the processed tensor back to a NumPy array for display
    # 1. Move to CPU if it's on GPU: .cpu()
    # 2. Convert to NumPy array: .numpy()
    # 3. Scale from [0, 1] to [0, 255] and convert to uint8: * 255).astype(np.uint8)
    display_img_np = (final_processed_tensor.cpu().numpy() * 255).astype(np.uint8)
    
    # Transpose from PyTorch's CHW (Channel, Height, Width) to Matplotlib's HWC (Height, Width, Channel)
    if display_img_np.ndim == 3: # If it's a 3D image (e.g., 3, 32, 512)
        display_img_np = np.transpose(display_img_np, (1, 2, 0))
        # If the image is grayscale but has 3 identical channels (due to .convert("RGB")),
        # reduce it to 1 channel for cleaner display with `cmap='gray'`.
        if display_img_np.shape[2] == 3 and np.all(display_img_np[:,:,0] == display_img_np[:,:,1]):
            display_img_np = display_img_np[:,:,0] # Take one channel
    
    plt.figure(figsize=(12, 5))

    # Display Original Image
    plt.subplot(1, 2, 1)
    plt.title("Ảnh Gốc")
    plt.imshow(original_image)
    plt.axis('off')

    # Display Processed Image
    plt.subplot(1, 2, 2)
    plt.title(f"Ảnh Đã Xử Lý ({display_img_np.shape[1]}x{display_img_np.shape[0]})")
    # Use cmap='gray' if the image is 2D (grayscale) or 3D with 1 channel
    if display_img_np.ndim == 2 or (display_img_np.ndim == 3 and display_img_np.shape[2] == 1):
        plt.imshow(display_img_np, cmap='gray')
    else:
        plt.imshow(display_img_np) # For RGB images
    plt.axis('off')
    plt.show()