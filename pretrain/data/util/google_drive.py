#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import requests
import pdb
from termcolor import colored


CHUNK_SIZE = 32768

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/u/1/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    
    print(colored(response.cookies.items(), 'red'))
    token=None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
