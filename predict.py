import argparse
import os
import shutil
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from models.model import DeepLabV3
# from pspnet import PSPNet
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from processor.TiffImageProcessor import TiffImageProcessor
from processor.ImageStitcher import ImageStitcher
from processor.ImageProcessor import ImageProcessor
from processor.change_shape import change_shape
from processor.down_image import MapDownloader
from pspnet_new import PSPNet
from processor.regularization import BuildingFootprintRegularization
from params import *

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

class Inference_Dataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.ids = os.listdir(os.path.join(self.root_dir))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,idx):
        id = self.ids[idx]
        image = Image.open(os.path.join(self.root_dir, id))
        image = self.transforms(image)
        return image,id

def reference():

    dataset = Inference_Dataset(root_dir=result+"/qietu", transforms=img_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=4)
    num_classes=2
    model = PSPNet(num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    tbar = tqdm(dataloader)
    model.eval()
    for image, id in tbar:
        predict = model(image)
        predict_list = torch.argmax(predict, dim=1).byte().numpy()  # n x h x w
        batch_size = predict_list.shape[0]  # batch大小
        for i in range(batch_size):
            png = predict_list[i]
            label_save_path = result + "/jieyi/" + id[i]
            image = Image.fromarray(png)
            image.save(label_save_path)
    
if __name__ == '__main__':
    if os.path.exists(result):
        shutil.rmtree(result)
    os.makedirs(result+"/qietu")
    os.makedirs(result+"/jieyi")    
    # 创建地图下载器实例并使用它下载地图瓦片并创建地理参考的GeoTIFF文件
    map_downloader = MapDownloader(url, min_lat, max_lat, min_lon, max_lon, zoom_level)
    result_image = map_downloader.download_tiles()
    print(map_downloader.deg2num3857(min_lon, min_lat))
    lon_min, lat_min, lon_max, lat_max = map_downloader.num2deg3857(*map_downloader.deg2num3857(min_lon, min_lat))
    lon_min1, lat_min1, lon_max1, lat_max1 = map_downloader.num2deg3857(*map_downloader.deg2num3857(max_lon, max_lat))
    map_downloader.create_georeferenced_tiff(result_image, lon_min, lat_min, lon_max1, lat_max1, image)
    processor = TiffImageProcessor(image, result+"/qietu", 512, 0)
    processor.tif_crop()
    processor = ImageProcessor()
    for file_name in os.listdir(result+"/qietu"):
        input_file = os.path.join(result+"/qietu", file_name)
        processor.compress(input_file, input_file)
    reference()
    regularization = BuildingFootprintRegularization(result + "/jieyi/")
    regularization.process_images()
    name,_ = os.path.splitext(image)
    stitcher = ImageStitcher(image,
                        result+"/jieyi",
                        name+"_predict.tif", 0)
    # Stitch images
    stitcher.tif_stitch()
    change_shape(input_raster = name+"_predict.tif")
    shutil.rmtree(result+"/qietu")
    shutil.rmtree(result+"/jieyi")
    os.remove(name+"_predict.tif")
    os.rmdir(result)