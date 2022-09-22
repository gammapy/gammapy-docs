import matplotlib.pyplot as plt
from astropy import units as u
from gammapy.image.models import SkyDisk

lons = np.linspace(0, 0.3, 500) * u.deg

r_0 = 0.2 * u.deg
edge = 0.1 * u.deg

disk = SkyDisk(lon_0="0 deg", lat_0="0 deg", r_0=r_0, edge=edge)
profile = disk(lons, 0 * u.deg)
plt.plot(lons, profile / profile.max(), alpha=0.5)
plt.xlabel("Radius (deg)")
plt.ylabel("Profile (A.U.)")

edge_min, edge_max = (r_0 - edge / 2.).value, (r_0 + edge / 2.).value
plt.vlines([edge_min, edge_max], 0, 1, linestyles=["--"])
plt.annotate("", xy=(edge_min, 0.5), xytext=(edge_min + edge.value, 0.5),
             arrowprops=dict(arrowstyle="<->", lw=2))
plt.text(0.2, 0.52, "Edge width", ha="center", size=12)
plt.hlines([0.95], edge_min - 0.02, edge_min + 0.02, linestyles=["-"])
plt.text(edge_min + 0.02, 0.95, "95%", size=12, va="center")
plt.hlines([0.05], edge_max - 0.02, edge_max + 0.02, linestyles=["-"])
plt.text(edge_max - 0.02, 0.05, "5%", size=12, va="center", ha="right")
plt.show()