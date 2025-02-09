import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
class preprocessing:
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        self.image = cv2.imread(self.in_path)

    def denoise_image(self):
        self.image = cv2.fastNlMeansDenoisingColored(self.image, None, 20, 20 , 7, 21)

    def enhance_contrast(self):
        yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        self.image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def restore_color(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1]*1.5
        self.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def super_resolution(self, model_path = "esrgan.pth"):
        device = torch.device("cpu")
        from 深度学习.model.arch import RRDBNet
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        h, w, _ = self.image.shape
        new_h, new_w = h // 4 * 4, w // 4 * 4
        image_resized = cv2.resize(self.image, (new_w, new_h))

        img = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img).squeeze(0).cpu()

        output = output * 0.5 + 0.5
        output = transforms.ToPILImage()(output)

        self.image = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

    def save_preprocessed_image(self):
        cv2.imwrite(out_path, self.image)

if __name__ == "__main__":
    in_path = "InPutPath..."
    out_path = "OutPutPath..."
    modelpath = "RRDB_PSNR_x4.pth" #模型的存储路径
    prepro_case = preprocessing(in_path, out_path)
    prepro_case.denoise_image()
    cv2.imshow("denoised", prepro_case.image)
    prepro_case.enhance_contrast()
    cv2.imshow("contrast_enhanced", prepro_case.image)
    prepro_case.restore_color()
    cv2.imshow("color_restored", prepro_case.image)
    prepro_case.super_resolution(modelpath)
    cv2.imshow("super_resolution", prepro_case.image)
    prepro_case.save_preprocessed_image()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
