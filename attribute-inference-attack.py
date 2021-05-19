# Python 3.8.5


import UTKFace
import Target
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


# transform = transforms.Compose(
#     [ transforms.Resize((356, 356)),
#       transforms.RandomCrop((299, 299)),
#       transforms.ToTensor(),
#       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

transform = transforms.Compose(
                    [ transforms.Resize(size=256),
                      transforms.CenterCrop(size=224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

test_set = UTKFace.UTKFace(root='UTKFace', test=True, transform=transform)
test_loader = DataLoader(dataset=test_set,
                         shuffle=True,
                         batch_size=32,
                         num_workers=1)


#target = Target.Target(device='cuda:2', train=True, ds_root='UTKFace')
target = Target.Target(device='cuda:2')
#print(f'Test Acc: {target.evaluate_model(target.model, test_loader)}')
image = Image.open('UTKFace/test/1_0_0_20161219154556757.jpg').convert("RGB")
image = transform(image)
image.to('cuda:2')

target.model.eval()
logits = target.model(image)
_, preds = torch.max(logits, dim=1)
print(preds)
