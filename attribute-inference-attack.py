# Python 3.8.5


import UTKFace
import Target
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


transform = transforms.Compose(
    [ transforms.Resize((356, 356)),
      transforms.RandomCrop((299, 299)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

test_set = UTKFace.UTKFace(root='UTKFace', test=True, transform=transform)
test_loader = DataLoader(dataset=test_set,
                         shuffle=True,
                         batch_size=32,
                         num_workers=1)


target = Target.Target()
target.evaluate_model(target.model, test_loader)
