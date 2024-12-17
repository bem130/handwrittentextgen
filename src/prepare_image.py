import fitz
from PIL import Image, ImageDraw, ImageFont
import os

dataset_dir = "font_data"
input_dir = "image"

def proc(dataid,output_dir,input):
    os.makedirs(os.path.join(dataset_dir,output_dir), exist_ok=True)
    # PDFファイルを開く
    doc = fitz.open(os.path.join(dataset_dir,input_dir,input))
    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            filepath = os.path.join(dataset_dir,input_dir,f"page_{i+1}_img_{img_index+1}.{image_ext}")
            with open(filepath, "wb") as img_file:
                print(filepath)
                img_file.write(image_bytes)

            # 画像を読み込む
            input_image = Image.open(filepath)

            # 切り出す領域を指定（左、上、右、下）
            inwidth, inheight = input_image.size
            height = 128
            width = 512
            x = 100
            y = 360
            lineh = 150
            slidex = 100
            linemax = 19
            for linen in range(linemax):
                sliden = 0
                while x+slidex*sliden+width < inwidth:
                    box = (x+slidex*sliden,y+lineh*linen,x+slidex*sliden+width,y+height+lineh*linen)
                    cropped_image = input_image.crop(box)
                    cropped_image.save(os.path.join(dataset_dir,output_dir,f"{dataid}-{linen}-{sliden}.{image_ext}"))
                    sliden+=1


if __name__ == "__main__":
    proc(0,"train/input","train-1-input.pdf")
    proc(0,"train/target","train-1-target.pdf")
