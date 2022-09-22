import numpy as np
import matplotlib.pyplot as plt
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import DiskSpatialModel

model = DiskSpatialModel("2 deg", "2 deg", "1 deg", 0.8, "30 deg", frame="galactic")

m_geom = WcsGeom.create(
    binsz=0.01, width=(3, 3), skydir=(2, 2), coordsys="GAL", proj="AIT"
)
coords = m_geom.get_coord()
vals = model(coords.lon, coords.lat)
skymap = Map.from_geom(m_geom, data=vals.value)

_, ax, _ = skymap.smooth("0.05 deg").plot()

transform = ax.get_transform("galactic")
ax.scatter(2, 2, transform=transform, s=20, edgecolor="red", facecolor="red")
ax.text(1.7, 1.85, r"$(l_0, b_0)$", transform=transform, ha="center")
ax.plot(
    [2, 2 + np.sin(np.pi / 6)],
    [2, 2 + np.cos(np.pi / 6)],
    color="r",
    transform=transform,
)
ax.vlines(x=2, color="r", linestyle="--", transform=transform, ymin=0, ymax=5)
ax.text(2.15, 2.3, r"$\phi$", transform=transform)

plt.show()