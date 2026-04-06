import torch
from torchvision import transforms

def basic_aug():
    '''
    Traditional Image Augmentation Pipeline.
    This creates ONE augmented output image per input by applying 
    multiple transformations in a specific sequence.
    '''
    return transforms.Compose([
        # Step 1: Horizontal Flip
        # p=0.5: There is a 50% probability that the image will be flipped.
        transforms.RandomHorizontalFlip(p=0.5), 
        
        # Step 2: Vertical Flip
        # p=0.5: There is a 50% probability that the image will be flipped upside down.
        transforms.RandomVerticalFlip(p=0.5),   
        
        # Step 3: Discrete Rotation
        # degrees=[...]: Randomly picks exactly one angle from this specific list.
        transforms.RandomRotation(degrees=(0, 270)),        
        
        # Step 4: Color Distortion
        # brightness, contrast, saturation=0.2: Randomly shifts these values 
        # within the range [0.8, 1.2] of the original intensity.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        
        # Step 5: Random Crop & Resize
        # scale=(0.8, 1.0): Randomly selects a portion of the image (80% to 100% area)
        # 224: Resizes that selected portion back to 224x224 pixels.
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        
        # Step 6: Tensor Conversion
        # Converts the PIL Image to a PyTorch Tensor and scales values to [0, 1].
        transforms.ToTensor(),
        
        # Step 7: Value Clipping
        # Lambda function: Ensures all pixels stay strictly between 0 and 1 
        # even if ColorJitter pushed them slightly out of bounds.
        transforms.Lambda(lambda img: torch.clamp(img, 0, 1)), 
        
        # Step 8: ImageNet Normalization
        # mean/std: Standardizes the tensor so its distribution matches 
        # what the pretrained ResNet50 model expects.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

