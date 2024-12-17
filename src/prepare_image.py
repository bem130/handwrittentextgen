import fitz
from PIL import Image, ImageDraw, ImageFont
import os
import random
import shutil

dataset_dir = "font_data"
input_dir = "image"

def delete_folder_contents(folder_path):
    # フォルダ内の全てのファイルとサブフォルダを削除
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)  # サブフォルダの場合は再帰的に削除
        else:
            os.remove(file_path)  # ファイルの場合は削除

os.makedirs(os.path.join(dataset_dir,"train/input"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir,"train/target"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir,"val/input"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir,"val/target"), exist_ok=True)

delete_folder_contents(os.path.join(dataset_dir,"train/input"))
delete_folder_contents(os.path.join(dataset_dir,"train/target"))
delete_folder_contents(os.path.join(dataset_dir,"val/input"))
delete_folder_contents(os.path.join(dataset_dir,"val/target"))


# 画像切り出しの調整
height = 128
width = 512
x = 100
y = 360
lineh = 150
slidex = 100
linemax = 19

def generate_random_array(l, p):
    return ["val" if random.random() < p else "train" for _ in range(l)]
datatype = generate_random_array(linemax,10/ 100)
print(datatype)

print(f"train: "+str(datatype.count("train")))
print(f"val  : "+str(datatype.count("val")))

def proc(dataid,output_dir,input):
    # PDFファイルを開く
    doc = fitz.open(os.path.join(dataset_dir,input_dir,input))
    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            filepath = os.path.join(dataset_dir,input_dir,f"{input_dir}.{image_ext}")
            with open(filepath, "wb") as img_file:
                img_file.write(image_bytes)

            # 画像を読み込む
            input_image = Image.open(filepath)

            # 切り出す領域を指定（左、上、右、下）
            inwidth, inheight = input_image.size
            for linen in range(linemax):
                sliden = 0
                while x+slidex*sliden+width < inwidth:
                    box = (x+slidex*sliden,y+lineh*linen,x+slidex*sliden+width,y+height+lineh*linen)
                    cropped_image = input_image.crop(box)
                    cropped_image.save(os.path.join(dataset_dir+"/"+datatype[linen]+"/"+output_dir,f"{dataid}-{linen}-{sliden}.{image_ext}"))
                    sliden+=1


proc(0,"input/","train-1-input.pdf")
proc(0,"target/","train-1-target.pdf")
