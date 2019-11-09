from pathlib import Path
import base64, sys, numpy as np

from bs4 import BeautifulSoup as BS
import bs4


import time
import os
import PIL.Image as Image
from IPython.display import display
path = Path(__file__).parent
result_html2 = path/'result2.html'
file='Sedan.html'
result_html2=path/'vendor'/'result_files'/file
result_html = str(result_html2.open().read())
print(result_html)
