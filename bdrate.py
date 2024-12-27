# This is for calcurating B-D rate for both methods
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

# data
proposed_x = np.array([78.43, 70.27, 63.54, 56.94, 50.66, 43.99, 37.28, 29.93, 22.73])
proposed_y = np.array([0.9872, 0.9869, 0.9867, 0.9864, 0.9861, 0.9855, 0.9848, 0.9835, 0.9807])

jpeg_x = np.array([109.77, 90.55, 49.74, 34.33, 27.15, 21.85, 17.66])
jpeg_y = np.array([0.9193, 0.9105, 0.9033, 0.8297, 0.7516, 0.5749, 0.3713])

common_range = np.linspace(min(min(proposed_x), min(jpeg_x)), max(max(proposed_x), max(jpeg_x)), 100)
proposed_interp = interp1d(proposed_x, proposed_y, kind='cubic', fill_value="extrapolate")
jpeg_interp = interp1d(jpeg_x, jpeg_y, kind='cubic', fill_value="extrapolate")

proposed_y_interp = proposed_interp(common_range)
jpeg_y_interp = jpeg_interp(common_range)

# simps method
proposed_integral = simps(proposed_y_interp, common_range)
jpeg_integral = simps(jpeg_y_interp, common_range)

# B-D Rate calcuration
bdrate = (proposed_integral - jpeg_integral) / jpeg_integral * 100

print(f"B-D Rate: {bdrate:.2f}%")
