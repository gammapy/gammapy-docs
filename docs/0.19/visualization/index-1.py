import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from gammapy.maps import Map
from gammapy.data import EventList
from gammapy.visualization import MapPanelPlotter

skydir = SkyCoord("0d", "0d", frame="galactic")
counts = Map.create(skydir=skydir, width=(180, 10), frame="galactic")

events = EventList.read("$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_events_selected.fits.gz")
counts.fill_events(events)

smoothed = counts.smooth("0.1 deg")

fig = plt.figure(figsize=(12, 6))
xlim, ylim = Angle(["90d", "270d"]), Angle(["-5d", "5d"])
plotter = MapPanelPlotter(figure=fig, xlim=xlim, ylim=ylim, npanels=3)

axes = plotter.plot(smoothed, vmax=10)
plt.show()