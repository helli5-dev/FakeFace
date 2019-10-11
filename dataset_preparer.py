from os import listdir
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from numpy import savez_compressed

model = MTCNN()

#Constants:
IMAGE_SIZE = (128, 128)
DATASET_SIZE = 50000
DATA_PATH = 'dataset/celeba/'
DATASET_NAME = 'celeba128.npz'

def loadImage(path):
    image = Image.open(path)
    image = image.convert('RGB')
    pixels = asarray(image)
    return pixels

def getFace(pixels):
    face = model.detect_faces(pixels)

    if (len(face) == 0):
        return None
    
    x, y, w, h = face[0]['box']
    x, y = abs(x), abs(y)

    facePixels = pixels[y: y + h, x: x + w]

    image = Image.fromarray(facePixels)
    image = image.resize(IMAGE_SIZE)

    return asarray(image)

def saveDataset():
    faces = list()

    for fileName in listdir(DATA_PATH):
        face = getFace(loadImage(DATA_PATH + fileName))
        
        if (face is None):
            continue
        
        faces.append(face)
        print(len(faces), face.shape)

        if (len(faces) >= DATASET_SIZE):
            break
    
    savez_compressed(DATASET_NAME, asarray(faces))

saveDataset()