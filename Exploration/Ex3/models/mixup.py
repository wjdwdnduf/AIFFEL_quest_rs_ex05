import torch
import torch.nn.functional as F
import numpy as np

def mixup_2_images(image_a, image_b, label_a, label_b, alpha=1.0, num_classes=120):
    '''
    Combines two images and their labels using linear interpolation (Mixup).

    Logic:
    1. Sample a mixing ratio (lam) from a Beta distribution.
    2. Convert integer labels into One-Hot float vectors to allow for fractional blending.
    3. Mix the pixels: (lam * image_a) + ((1 - lam) * image_b).
    4. Mix the labels: (lam * label_a) + ((1 - lam) * label_b).

    Args:
        image_a, image_b: Individual image tensors [C, H, W].
        label_a, label_b: Corresponding class labels (int or tensor).
        alpha: Beta distribution parameter (controls the strength of mixing).
        num_classes: Total number of categories for One-Hot encoding.

    Returns:
        mixed_image: The blended "ghost" image.
        mixed_label: The blended soft-target label.
    '''
    # Step 1: Sample the mixing weight (lambda)
    # alpha=1.0 results in a uniform distribution between 0 and 1.
    lam = np.random.beta(alpha, alpha) # REPLACED: ratio = torch.rand(1).item() with Standard Beta distribution sampling

    # Step 2: Convert labels to One-Hot if they are indices
    # We need floats to perform weighted addition (e.g., 0.4*Class1 + 0.6*Class2).
    if not isinstance(label_a, torch.Tensor):
        label_a = torch.tensor(label_a)
    if not isinstance(label_b, torch.Tensor):
        label_b = torch.tensor(label_b)
    if not torch.is_tensor(label_a) or label_a.dim() == 0:
        label_a = F.one_hot(torch.tensor(label_a).long(), num_classes=num_classes).float()
    if not torch.is_tensor(label_b) or label_b.dim() == 0:
        label_b = F.one_hot(torch.tensor(label_b).long(), num_classes=num_classes).float()

    # Step 3: Linearly interpolate the pixel values
    # This creates a "transparency" effect where both images are visible.
    mixed_image = (lam * image_a) + ((1 - lam) * image_b)

    # Step 4: Linearly interpolate the labels
    mixed_label = (lam * label_a) + ((1 - lam) * label_b)

    return mixed_image, mixed_label

def mixup(images, labels, prob=1.0, batch_size=16, img_size=224, num_classes=120):
    '''
    Applies the Mixup augmentation strategy to an entire batch of data.

    Logic:
    1. Iterate through the batch and decide whether to apply Mixup based on 'prob'.
    2. For each image, pick a random partner image from the same batch.
    3. Call mixup_2_images to blend the pair.
    4. Stack the results into consistent batch tensors for training.

    Args:
        images: Batch of images [Batch, C, H, W].
        labels: Batch of class indices.
        prob: Probability of applying Mixup to a sample.
        batch_size: Number of samples in the batch.
        img_size: Spatial dimension of the images.
        num_classes: Total number of categories.

    Returns:
        mixed_imgs: Augmented image batch tensor.
        mixed_labels: Soft-target label batch tensor.
    '''
    mixed_imgs = []
    mixed_labels = []

    # Step 1: Process each sample in the batch
    for i in range(batch_size):
        image_a, label_a = images[i], labels[i]

        # Step 2: Probability check
        if np.random.random() < prob:
            # Pick a random partner image (image_b) from the same batch
            j = torch.randint(0, batch_size, (1,)).item()
            image_b, label_b = images[j], labels[j]

            # Step 3: Apply the 2-image mixing logic
            m_img, m_lbl = mixup_2_images(image_a, image_b, label_a, label_b, alpha=1.0, num_classes=num_classes)
            
            mixed_imgs.append(m_img)
            mixed_labels.append(m_lbl)
        else:
            # Step 4: If skipping, still convert label to One-Hot for batch consistency
            mixed_imgs.append(image_a)
            mixed_labels.append(F.one_hot(torch.tensor(label_a).long(), num_classes=num_classes).float())

    # Step 5: Stack into final batch tensors
    mixed_imgs = torch.stack(mixed_imgs)
    mixed_labels = torch.stack(mixed_labels)

    return mixed_imgs, mixed_labels

