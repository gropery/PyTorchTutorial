from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'data/train/ants_image/0013035.jpg'
img = Image.open(img_path)
print(type(img))

writer = SummaryWriter('logs')

tensor_trans = transforms.ToTensor()
# tensor_img = tensor_trans(img)调用call方法
# 相当于tensor_img = tensor_trans.__call__(img)
tensor_img = tensor_trans(img)

writer.add_image('Tensor_img', tensor_img)

writer.close()


