from torchvision import transforms
from torchvision.transforms import InterpolationMode


class SimCLRTrainTransforms:
    def __init__(
        self,
        img_size=224,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=InterpolationMode.BICUBIC,
        p_flip=0.5,
        p_jitter=0.8,
        p_drop=0.2,
        kernel_size=(23, 23),
        sigma=(0.1, 2.0)
    ):
        self.online_transform = SimCLROnlineTransform()
        self.target_transform = SimCLRTargetTransform()
    
    def __call__(self, sample):
        return self.online_transform(sample), self.target_transform(sample)


class SimCLREvalTransform:
    def __init__(self):
        self.transform = None
    
    def __call__(self, sample):
        return self.transform(sample)


class SimCLROnlineTransform:
    def __init__(
        self,
        img_size=224,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=InterpolationMode.BICUBIC,
        p_flip=0.5,
        p_jitter=0.8,
        p_drop=0.2,
        kernel_size=(23, 23),
        sigma=(0.1, 2.0)
    ):
        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1
        )
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio, interpolation=interpolation),
            transforms.RandomHorizontalFlip(p_flip),
            transforms.RandomApply([color_jitter], p=p_jitter),
            transforms.Grayscale(p_drop),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
            transforms.ToTensor()
        ])
    
    def __call__(self, sample):
        return self.transform(sample)


class SimCLRTargetTransform:
    def __init__(
        self,
        img_size=224,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=InterpolationMode.BICUBIC,
        p_flip=0.5,
        p_jitter=0.8,
        p_drop=0.2,
        kernel_size=(23, 23),
        sigma=(0.1, 2.0)
    ):
        color_jitter = transforms.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.8
        )
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio, interpolation=interpolation),
            transforms.RandomHorizontalFlip(p_flip),
            transforms.RandomApply(color_jitter, p=p_jitter),
            transforms.Grayscale(p_drop),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
            transforms.ToTensor()
        ])
    
    def __call__(self, sample):
        return self.transform(sample)