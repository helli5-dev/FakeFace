from numpy import load
from matplotlib import pyplot

dataset = load('celeba128.npz')
faces = dataset['arr_0']

for i in range(10 * 10):
    pyplot.subplot(10, 10, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(faces[i].astype('uint8'))
    print(i)

pyplot.show()