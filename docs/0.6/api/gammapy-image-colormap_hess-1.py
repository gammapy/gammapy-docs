from gammapy.image import colormap_hess, illustrate_colormap
import matplotlib.pyplot as plt
cmap = colormap_hess()
illustrate_colormap(cmap)
plt.show()