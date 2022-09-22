from gammapy.image import colormap_milagro, illustrate_colormap
import matplotlib.pyplot as plt
cmap = colormap_milagro()
illustrate_colormap(cmap)
plt.show()