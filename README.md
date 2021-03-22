# Photo-Id API
Developted por Bruno Leão

### Installation

Aplication requires python 3 to run.

Install the dependencies and start the server.

```sh
$ pip3 install -r requirements.txt
```

### Start

```sh
$ python3 app.py
```

# Usage
## Routes:
Using postman or insonmia or another provider to test it, follow the instructions to each endpoint.

### 1 - http://127.0.0.1:5000/predict-species - POST
Route to indentify in image the species of a whale between two species: Humpback or Right whale 
##### params:
- image1: Image of the cetacean, in png or jpg format.

##### response: 
The class of the image, in this case: Humpback whale or Right whale

### 2 - http://127.0.0.1:5000/identify 
Route to indentify in an image, the individual. Only works for Humpback, feel free to make it work with right whale.

##### params:
- image1: Image of the humpback whale, in png or jpg format.

##### response: 
The class of the image, in this case it will be the name of the individual in the picture

### 3 - http://127.0.0.1:5000/identify-between-individuals
The funcionality is the same of the other, but with a difference. The user can input a list of individuals that will be considered as a possible response. So only between these names, the identification will be made.

##### params:
- image1: Image of the humpback whale, in png or jpg format.
- individuals: String with name of possible individuals, example: "wd2908,we1329,we1342,..."

##### response: 
The class of the image, in this case it will be the name of the individual in the picture

### 4 - http://127.0.0.1:5000/compare-images
As the name says, this route is to check in two different images whether they represent the same individual or not.

##### params:
- image1: Image of the humpback whale, in png or jpg format.
- image2: Image of the humpback whale, in png or jpg format.

##### response: 
A simple message saying if they are equal or not.

### 5 - http://127.0.0.1:5000/update-base
The last route is to update the information in the base. All individuals used to train and test are saved in npy arrays and they are used to identify new images. At this enpoint, inputting the image and the correct class of the image (Could be the one identified or the one that the user is confident is the correct one ) will add these image to the database, helping to improve the Photo-ID of the neural network.

##### params:
- image1: Image of the humpback whale, in png or jpg format.
- classe: The name of the individual 

##### response: 
A simple message saying if it went everything okay or something went wrong.

## What's next?

Since this is an open source api, I highly encourage and desire to see contributions to this project and these could be:
- New species in the photo-id
- A Neural Network to the Right Whale
- Enhance the current networks
- Any other that you feel is going to be good for the project, feel free.
 
### Any doubt please contact me.


