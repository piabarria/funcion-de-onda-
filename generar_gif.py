import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob

files = sorted(glob.glob("output_*.dat"))

images = []

for file in files:
    data = np.loadtxt(file)

    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    N = int(np.sqrt(len(z)))
    Z = z.reshape((N,N))

    plt.imshow(Z, origin='lower')
    plt.colorbar()
    plt.title(file)

    filename = file.replace(".dat", ".png")
    plt.savefig(filename)
    plt.close()

    images.append(imageio.imread(filename))

imageio.mimsave("wave.gif", images, duration=0.1)
