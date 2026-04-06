import torch
import numpy as np
import torch.nn.functional as F

def rand_bbox(size, alpha=1.0):
    '''
    Generates a random bounding box coordinates based on a Beta distribution.

    Logic:
    1. Sample a combination ratio (lambda) from a Beta distribution.
    2. Calculate the patch width and height using the square root of (1 - lambda) 
       to ensure the area ratio is preserved.
    3. Randomly select a center point (cx, cy) within the image dimensions.
    4. Calculate the corner coordinates (x_min, y_min, x_max, y_max) and clip 
       them so they don't extend outside the image boundaries.

    Args:
        size: The shape of the image (Batch, Channels, Height, Width)
        alpha: Hyperparameter for Beta distribution (usually 1.0)
        
    Returns:
        x_min, y_min, x_max, y_max: Coordinates for the cut-out region
    '''
    W = size[2] # Image Width (from shape C, H, W)
    H = size[3] # Image Height

    # Step 1: Sample lambda (lam). alpha=1.0 makes this a Uniform distribution.
    lam = np.random.beta(alpha, alpha)

    # Step 2: Calculate patch dimensions.
    # We use sqrt so that: (cut_w * cut_h) / (W * H) is approximately (1 - lam).
    cut_rat = np.sqrt(1. - lam) # the patch area is proportional to $\lambda$ sampled from a Beta distribution. 
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Step 3: Pick a random center point for the patch.
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Step 4: Compute bounding box corners and clip to image size.
    # // 2 ensures the center (cx, cy) is roughly in the middle of the patch.
    x_min = np.clip(cx - cut_w // 2, 0, W)
    y_min = np.clip(cy - cut_h // 2, 0, H)
    x_max = np.clip(cx + cut_w // 2, 0, W)
    y_max = np.clip(cy + cut_h // 2, 0, H)

    return x_min, y_min, x_max, y_max

def mix_2_images(image_a, image_b, alpha=1.0):
    '''
    Combines two images by cutting a patch from image_b and pasting it onto image_a.
    Also calculates the final lambda (ratio) based on the actual pixels replaced.

    Logic:
    1. Get the bounding box coordinates from the rand_bbox function.
    2. Clone image_a to ensure the original data isn't modified in-place.
    3. Use PyTorch slicing to replace the pixels in the bounding box area 
       with pixels from the same location in image_b.
    4. Re-calculate the actual lambda. Since we clip the box at the image edges, 
       the final area ratio might differ slightly from the initial lambda.
    
    Args:
        image_a: The base image (Tensor shape [C, H, W])
        image_b: The image providing the patch (Tensor shape [C, H, W])
        alpha: Distribution parameter for rand_bbox
        
    Returns:
        mixed_img: The resulting augmented image
        lam: The final ratio of image_a remaining in the mix (used for label mixing)
    '''
    # We add a fake batch dimension [1, C, H, W] to size for rand_bbox compatibility
    # or simply pass image_a.shape expanded if working with single images
    size = [1] + list(image_a.shape) 
    
    # Step 1: Get random coordinates
    x_min, y_min, x_max, y_max = rand_bbox(size, alpha)
    
    # Step 2: Create a copy to avoid modifying the original image_a
    mixed_img = image_a.clone()
    
    # Step 3: Paste the patch from image_b into image_a
    mixed_img[:, y_min:y_max, x_min:x_max] = image_b[:, y_min:y_max, x_min:x_max]
    
    # Step 4: Calculate the actual lambda based on the size of the patch
    # Important because clipping near edges changes the actual area used
    actual_patch_area = (x_max - x_min) * (y_max - y_min)
    total_area = image_a.shape[1] * image_a.shape[2]
    lam = 1 - (actual_patch_area / total_area)
    
    return mixed_img, lam

def mix_2_labels(label_a, label_b, lam, num_classes=120):
    '''
    Blends two labels together into a single soft-label vector based on the area ratio.

    Logic:
    1. Check if input labels are integers; if so, convert them to One-Hot float tensors.
    2. Apply the lambda (lam) to label_a and (1 - lam) to label_b.
    3. Sum the weighted vectors to create a distribution where the model learns 
       both classes simultaneously relative to their visual presence.

    Args:
        label_a: Ground truth label for the base image.
        label_b: Ground truth label for the patch image.
        lam: The calculated ratio of image_a pixels remaining in the mix.
        num_classes: Total number of categories in the dataset.

    Returns:
        mixed_label: A soft-target tensor of shape [num_classes].
    '''
    # Step 1: Convert integer indices to One-Hot vectors (Float)
    # This is necessary because we cannot "blend" two integers (e.g., class 3 and 7) 
    # without creating a float distribution.
    if not isinstance(label_a, torch.Tensor):
        label_a = torch.tensor(label_a)
    if not isinstance(label_b, torch.Tensor):
        label_b = torch.tensor(label_b)
    if not torch.is_tensor(label_a) or label_a.dim() == 0:
        label_a = F.one_hot(torch.tensor(label_a).long(), num_classes=num_classes).float()
    if not torch.is_tensor(label_b) or label_b.dim() == 0:
        label_b = F.one_hot(torch.tensor(label_b).long(), num_classes=num_classes).float()

    # Step 2: Calculate the weighted sum of both labels
    # If lam = 0.7, the output is 70% of Class A and 30% of Class B.
    mixed_label = (lam * label_a) + ((1 - lam) * label_b)
    
    return mixed_label


def cutmix(images, labels, prob=1.0, batch_size=16, img_size=224, num_classes=120):
    '''
    Applies the CutMix augmentation strategy to an entire batch of data.

    Logic:
    1. Iterate through the batch. For each sample, decide whether to apply CutMix 
       based on the probability 'prob'.
    2. If applying: Randomly select a partner image from the same batch, mix the 
       pixels using mix_2_images, and mix the labels using mix_2_labels.
    3. If skipping: Keep the original image and convert its label to a One-Hot 
       vector to maintain tensor shape consistency across the batch.
    4. Stack the individual results back into a single 4D image tensor and 2D label tensor.

    Args:
        images: Batch of images [Batch, C, H, W].
        labels: Batch of class indices.
        prob: Probability of applying CutMix (0.0 to 1.0).
        batch_size: Number of samples in the current batch.
        img_size: Spatial dimension (Height/Width) of the images.
        num_classes: Total number of categories for One-Hot encoding.

    Returns:
        mixed_imgs: Augmented image batch tensor.
        mixed_labels: Soft-target label batch tensor.
    '''
    mixed_imgs = []
    mixed_labels = []

    # Step 1: Process each image in the batch individually
    for i in range(batch_size):
        image_a, label_a = images[i], labels[i]

        # Step 2: Use random probability to decide if we apply CutMix
        if np.random.random() < prob:
            # Pick a random partner (image_b) from the same batch
            j = torch.randint(0, batch_size, (1,)).item()
            image_b, label_b = images[j], labels[j]

            # Step 3: Mix the pixels and extract the final area ratio (lam)
            m_img, lam = mix_2_images(image_a, image_b, alpha=1.0)
            
            # Step 4: Mix the labels according to that specific area ratio
            m_lbl = mix_2_labels(label_a, label_b, lam, num_classes)

            mixed_imgs.append(m_img)
            mixed_labels.append(m_lbl)
        else:
            # Step 5: If not mixing, we must still convert labels to One-Hot
            # This ensures the final 'mixed_labels' tensor has consistent shape.
            mixed_imgs.append(image_a)
            mixed_labels.append(F.one_hot(torch.tensor(label_a).long(), num_classes=num_classes).float())

    # Step 6: Combine the list of tensors into a single batch tensor
    # .reshape() ensures the dimensions match (Batch, Channel, Height, Width)
    mixed_imgs = torch.stack(mixed_imgs)
    mixed_labels = torch.stack(mixed_labels)

    return mixed_imgs, mixed_labels