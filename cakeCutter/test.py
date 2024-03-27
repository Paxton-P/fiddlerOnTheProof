import matplotlib.pyplot as plt
import pandas as pd

point_data = pd.read_csv('center_points1.csv', names=['x', 'y', 'loss'])

plt.scatter(point_data['x'], point_data['y'])
plt.xlim(0,20)
plt.ylim(0,10)
plt.show()
