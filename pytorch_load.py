import torch
import cv2
from torchvision import transforms
import Image
import torch.nn.functional as F
import datetime

data_transforms =transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
labels = {0:'ants',
          1:'apples',
          2:'bees',
          3:'bottles',
          4:'elephants',
          5:'grapes',
          6:'mango',
          7:'phones',
          8:'rabbits',
          9:'tv',
          10:'watermelon'}

device = torch.device('cpu')
#由于我的resnet50 fine-tuning了11种物体，是在GPU上训练的，所以下面加载模型的时候，必须映射设备类型到cpu上
model_ft = torch.load("best_11cls_resnet50.pkl",map_location=lambda storage,loc:storage)
model_ft = model_ft.to(device)
model_ft.eval()

def real_time_classification():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, frame = cam.read()
        cv2.imshow("webcam",frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        input_img = data_transforms(pil_img).reshape([1,3,224,224]).to(device)
        #input_img = input_img.reshape([1,3,224,224])
        #input_img = input_img.to(device)
        out = model_ft(input_img)
        softmax_result = F.softmax(out)
        top1_prob,top1_label = torch.topk(softmax_result,1)
        print(top1_prob,labels.get(int(top1_label)))
        cv2.waitKey(1)
    cv2.destroyAllWindows()

real_time_classification()