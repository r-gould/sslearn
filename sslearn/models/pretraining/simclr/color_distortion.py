from torchvision import transforms

def color_distortion(s: float = 1.0):

    #color_jitter = transforms.ColorJitter(*[0.8*s]*4)
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    return transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])