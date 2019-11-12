import shutil, os
from pathlib import Path
path = os.getcwd()
img_path = path+"\\saved_image.png"
temp= "\\tmp\\saved_image.png"
dest_path = path+"\\vendor\\result_files"
# shutil.copy(img_path, dest_path)
os.rename(dest_path+"\\saved_image.png", 'image.jpg')