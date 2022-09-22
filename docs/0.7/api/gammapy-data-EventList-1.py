import matplotlib.pyplot as plt
from gammapy.data import DataStore

ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
events = ds.obs(obs_id=23523).events
events.plot_time_map()
plt.show()