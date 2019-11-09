import bs4
import os
from pathlib import Path
path = Path(__file__).parent
result_html2 = path/'result2.html'
with open(result_html2) as inf:
    txt = inf.read()
    soup = bs4.BeautifulSoup(txt,features="html.parser")
    # create new link
    span = soup.find("span", {"id": "result1"})
    span.string.replace_with("New")
with open("result.html", "w") as file:
    file.write(str(soup))