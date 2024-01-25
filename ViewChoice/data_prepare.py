from xml.etree import ElementTree as ET
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import cv2
from PIL import Image

os.environ['PYTHONIOENCODING'] = 'utf-8'


def xml_to_json(xml_file, json_file):
    # 读取公司提供的xml文件  春笋  南沙
    tree = ET.parse(xml_file)
    root = tree.getroot()
    Blocks = root.find('Block')
    Photogroups = Blocks.find('Photogroups')
    Photogroup = Photogroups.findall('Photogroup')
    images_info = []
    for _ in Photogroup:
        Photos = _.findall('Photo')
        for ele in Photos:
            image_info = {}
            image_info['Group-PrincipalPoint'] = [np.float64(_.find('PrincipalPoint/x').text),
                                                  np.float64(_.find('PrincipalPoint/y').text)]
            image_info['Group-CameraOrientation'] = _.find('CameraOrientation').text
            image_info['Group-Distortion'] = [np.float64(_.find('Distortion/K1').text),
                                              np.float64(_.find('Distortion/K2').text),
                                              np.float64(_.find('Distortion/K3').text),
                                              np.float64(_.find('Distortion/P1').text),
                                              np.float64(_.find('Distortion/P2').text)]
            image_info['Group-FocalLength'] = np.float64(_.find('FocalLength').text)
            image_info['Group-SensorSize'] = np.float64(_.find('SensorSize').text)
            image_info['photo-Id'] = int(ele.find('Id').text)
            # should change image root path
            image_info['photo-ImagePath'] = 'F:/公司数据/南沙/01.原始照片' + ele.find('ImagePath').text[23:]
            image_info['photo-Rotation'] = [[np.float64(ele.find('Pose/Rotation/M_00').text),
                                             np.float64(ele.find('Pose/Rotation/M_01').text),
                                             np.float64(ele.find('Pose/Rotation/M_02').text)],
                                            [np.float64(ele.find('Pose/Rotation/M_10').text),
                                             np.float64(ele.find('Pose/Rotation/M_11').text),
                                             np.float64(ele.find('Pose/Rotation/M_12').text)],
                                            [np.float64(ele.find('Pose/Rotation/M_20').text),
                                             np.float64(ele.find('Pose/Rotation/M_21').text),
                                             np.float64(ele.find('Pose/Rotation/M_22').text)]]
            image_info['photo-Center'] = [np.float64(ele.find('Pose/Center/x').text),
                                          np.float64(ele.find('Pose/Center/y').text),
                                          np.float64(ele.find('Pose/Center/z').text)]
            image_info['photo-MetadataCenter'] = [np.float64(ele.find('Pose/Metadata/Center/x').text),
                                                  np.float64(ele.find('Pose/Metadata/Center/y').text),
                                                  np.float64(ele.find('Pose/Metadata/Center/z').text)]
            image_info['photo-ImageDimensions'] = [int(ele.find('ExifData/ImageDimensions/x').text),
                                                   int(ele.find('ExifData/ImageDimensions/y').text)]
            image_info['photo-GPS'] = [np.float64(ele.find('ExifData/GPS/Latitude').text),
                                       np.float64(ele.find('ExifData/GPS/Longitude').text),
                                       np.float64(ele.find('ExifData/GPS/Altitude').text)]
            image_info['photo-FocalLength'] = np.float64(ele.find('ExifData/FocalLength').text)
            image_info['photo-FocalLength35mmEq'] = np.float64(ele.find('ExifData/FocalLength35mmEq').text)
            image_info['photo-YawPitchRoll'] = [np.float64(ele.find('ExifData/YawPitchRoll/x').text),
                                                np.float64(ele.find('ExifData/YawPitchRoll/y').text),
                                                np.float64(ele.find('ExifData/YawPitchRoll/z').text)]
            images_info.append(image_info)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(images_info, f, ensure_ascii=False)


def xml_to_json_l7(xml_file, json_file):
    # 读取公司提供的xml文件  L7
    tree = ET.parse(xml_file)
    root = tree.getroot()
    Blocks = root.find('Block')
    Photogroups = Blocks.find('Photogroups')
    Photogroup = Photogroups.findall('Photogroup')
    images_info = []
    for _ in Photogroup:
        Photos = _.findall('Photo')
        for ele in Photos:
            image_info = {}
            image_info['Group-PrincipalPoint'] = [np.float64(_.find('PrincipalPoint/x').text),
                                                  np.float64(_.find('PrincipalPoint/y').text)]
            image_info['Group-CameraOrientation'] = _.find('CameraOrientation').text
            image_info['Group-Distortion'] = [np.float64(_.find('Distortion/K1').text),
                                              np.float64(_.find('Distortion/K2').text),
                                              np.float64(_.find('Distortion/K3').text),
                                              np.float64(_.find('Distortion/P1').text),
                                              np.float64(_.find('Distortion/P2').text)]
            image_info['Group-FocalLengthPixels'] = np.float64(_.find('FocalLengthPixels').text)
            image_info['photo-Id'] = int(ele.find('Id').text)
            # should change image root path
            image_info['photo-ImagePath'] = 'F:/公司数据/L7/01SourceImages/images' + ele.find('ImagePath').text[1:]
            image_info['photo-Rotation'] = [[np.float64(ele.find('Pose/Rotation/M_00').text),
                                             np.float64(ele.find('Pose/Rotation/M_01').text),
                                             np.float64(ele.find('Pose/Rotation/M_02').text)],
                                            [np.float64(ele.find('Pose/Rotation/M_10').text),
                                             np.float64(ele.find('Pose/Rotation/M_11').text),
                                             np.float64(ele.find('Pose/Rotation/M_12').text)],
                                            [np.float64(ele.find('Pose/Rotation/M_20').text),
                                             np.float64(ele.find('Pose/Rotation/M_21').text),
                                             np.float64(ele.find('Pose/Rotation/M_22').text)]]
            image_info['photo-Center'] = [np.float64(ele.find('Pose/Center/x').text),
                                          np.float64(ele.find('Pose/Center/y').text),
                                          np.float64(ele.find('Pose/Center/z').text)]
            image_info['photo-MetadataCenter'] = [np.float64(ele.find('Pose/Metadata/Center/x').text),
                                                  np.float64(ele.find('Pose/Metadata/Center/y').text),
                                                  np.float64(ele.find('Pose/Metadata/Center/z').text)]
            image_info['photo-ImageDimensions'] = [int(ele.find('ExifData/ImageDimensions/x').text),
                                                   int(ele.find('ExifData/ImageDimensions/y').text)]
            image_info['photo-GPS'] = [np.float64(ele.find('ExifData/GPS/Latitude').text),
                                       np.float64(ele.find('ExifData/GPS/Longitude').text),
                                       np.float64(ele.find('ExifData/GPS/Altitude').text)]
            image_info['photo-FocalLength'] = np.float64(ele.find('ExifData/FocalLength').text)
            image_info['photo-YawPitchRoll'] = [np.float64(ele.find('ExifData/YawPitchRoll/x').text),
                                                np.float64(ele.find('ExifData/YawPitchRoll/y').text),
                                                np.float64(ele.find('ExifData/YawPitchRoll/z').text)]
            images_info.append(image_info)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(images_info, f, ensure_ascii=False)


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    # xml_to_json_l7('./DataProcess/xmls/L7.xml', './DataProcess/xmls/L7Info.json')
    # xml_to_json('./DataProcess/xmls/ChunSun.xml', './DataProcess/xmls/ChunsunInfo.json')
    # DataProcess = read_json('./DataProcess/xmls/L7Info.json')
    # image_path = DataProcess[778]['photo-ImagePath']
    # img = plt.imread(image_path)
    # plt.imshow(img, origin='lower')
    # plt.show()
    print('This is data_prepare.py...')
