from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms


writer = SummaryWriter('logs')
img = Image.open('dataset/train/ants/0013035.jpg')
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTenser', img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm)
print(img_norm[0][0][0])

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image('Resize', img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(52)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)

writer.close()
