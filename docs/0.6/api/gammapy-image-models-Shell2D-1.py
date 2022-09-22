import numpy as np
import matplotlib.pyplot as plt
from gammapy.image.models import Shell2D

shell = Shell2D(amplitude=100, x_0=25, y_0=25, r_in=10, width=5)
y, x = np.mgrid[0:50, 0:50]
plt.imshow(shell(x, y), origin='lower', interpolation='none')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.colorbar(label='Brightness (A.U.)')
plt.grid(False)
plt.show()