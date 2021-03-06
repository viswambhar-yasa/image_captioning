## Image captioning using reinforcement learning
### Policy to actor method on deep convolution and recurrent networks
#### Project Seminar for artifical intelligence WS2021-22
##### Authors : Viswambhar Yasa, Venkata Mukund
## This file contains functions required to extract data from url
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def downloading_extraction(link, extraction_path='.'):
    """
    Args:
        link (str): url link 
        extraction_path (str, optional): path where data should be downloaded. Defaults to '.'.
    """
    url = urlopen(link)
    zipfile = ZipFile(BytesIO(url.read()))
    zipfile.extractall(path=extraction_path)


if __name__ == "__main__":
    images_link = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
    downloading_extraction(images_link)
    text_link = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    downloading_extraction(text_link, r'.\Flicker8k_Dataset\text_files')
