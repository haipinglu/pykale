import torch
from torchvision import transforms


def get_transform(kind, modality):
    """
    Define transforms (for commonly used datasets)

    Args:
        kind ([type]): the dataset (transformation) name
        modality (string): image type (RGB or Optical Flow)
    """

    if kind in ["epic", "gtea", "adl", "kitchen"]:
        if modality == 'rgb':
            transform = {
                'train': transforms.Compose([
                    ImglistToTensor(),
                    transforms.Resize(size=256),
                    transforms.RandomCrop(size=224),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    TensorPermute(),
                ]),
                'valid': transforms.Compose([
                    ImglistToTensor(),
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    TensorPermute(),
                ]),
                'test': transforms.Compose([
                    ImglistToTensor(),
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    TensorPermute(),
                ])
            }
        elif modality == 'flow':
            transform = {
                'train': transforms.Compose([
                    ImglistToTensor(),
                    transforms.Resize(size=256),
                    transforms.RandomCrop(size=224),
                    transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                    TensorPermute(),
                ]),
                'valid': transforms.Compose([
                    ImglistToTensor(),
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                    TensorPermute(),
                ]),
                'test': transforms.Compose([
                    ImglistToTensor(),
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                    TensorPermute(),
                ])
            }


    else:
        raise ValueError(f"Unknown transform kind '{kind}'")
    return transform


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``kale.loaddata.videos.VideoFrameDataset``.
    """

    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size `` NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """

        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])


class TensorPermute(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) to
    a torch.FloatTensor of shape (CHANNELS x NUM_IMAGES x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(1, 0, 2, 3).contiguous()