#!/usr/bin/env python3

'''\
Create art dataset.
'''

# Web stuff
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

import re
import os

fake_headers = {'User-Agent': UserAgent().random}
domain = 'https://www.wikiart.org'

# Artists
artists = [
  ("Gustave Caillebotte",
      "https://www.wikiart.org/en/gustave-caillebotte/all-works/text-list" ),
  ("Claude Monet",
      "https://www.wikiart.org/en/claude-monet/all-works/text-list" ),
  ("Edouard Manet",
      "https://www.wikiart.org/en/edouard-manet/all-works/text-list" ),
  ("Caravaggio",
      "https://www.wikiart.org/en/caravaggio/all-works/text-list" ),
  ("Alfred Sisley",
      "https://www.wikiart.org/en/alfred-sisley/all-works/text-list" ),
  ("Armand Guillaumin",
      "https://www.wikiart.org/en/armand-guillaumin/all-works/text-list" ),
  ("Paul Signac",
      "https://www.wikiart.org/en/paul-signac/all-works/text-list" ),
  ("Vincent van Gogh",
      "https://www.wikiart.org/en/vincent-van-gogh/all-works/text-list" ),
  ("M.C. Escher",
      "https://www.wikiart.org/en/m-c-escher/all-works/text-list" ),
  ]

# Image link pattern
image_link_pattern = re.compile('https[^"!]*jpg[^!]') # Select original pic


def save_image(artist, link):

  # Paths
  directory = os.path.join('art', artist)
  fileName = link.split('/')[-1]
  imgPath = os.path.join(directory, fileName)
  
  # Return if already downloaded
  if os.path.exists(imgPath):
    print('> Skip: ', imgPath)
    return

  # Fetch
  img_return = requests.get(link, stream=True)
  img_return.raise_for_status()

  # Save
  if not os.path.exists(directory):
    os.makedirs(directory)
  with open(imgPath, 'wb') as imgFile:
    imgFile.write(img_return.content)
  
  print('> ', imgPath)


def main():
  
  # Artists
  for artist in artists:

    # Artist page: get
    artist_page = requests.get(artist[1], headers=fake_headers)
    artist_page.raise_for_status()

    # Artist page: parse
    artist_page = BeautifulSoup(artist_page.text, 'lxml')
    paint_elems = artist_page.find_all('li', {'class': 'painting-list-text-row'})

    # Paintings
    for elem in paint_elems:

      elem = elem.find('a')
      painting_url = domain + elem.attrs['href']

      # Skip heuristic
      filePath = os.path.join('art', artist[0],
          painting_url.split('/')[-1]+'.jpg')
      if os.path.exists(filePath):
        print('> Skip: ', filePath)
        continue

      # Painting page: get
      painting_page = requests.get(painting_url, headers=fake_headers)
      painting_page.raise_for_status()

      # Painting page: parse
      painting_page = BeautifulSoup(painting_page.text, 'lxml')
      links_block = painting_page.find('main',
          {'ng-controller': 'ArtworkViewCtrl'}).attrs['ng-init']

      # Extract url of the original image
      image_link = image_link_pattern.findall(links_block)
      if not image_link: continue
      image_link = image_link[0][:-1]

      # Save
      save_image(artist[0], image_link)


if __name__ == '__main__':
  main()
