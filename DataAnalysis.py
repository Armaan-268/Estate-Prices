import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('dataset.csv')

# Plot_1: Age vs Price
x = dataset.iloc[1:, 2].values
y = dataset.iloc[1:, -1].values
plt.scatter(x,y)
plt.title('Age vs Price')
plt.xlabel('Age (years)')
plt.ylabel('Price (per unit area)')
plt.show()

# Plot_2: Distance from MRT Station vs Price
x = dataset.iloc[1:, 3].values
y = dataset.iloc[1:, -1].values
plt.scatter(x,y)
plt.title('Distance from MRT vs Price')
plt.xlabel('Distance from MRT')
plt.ylabel('Price (per unit area)')
plt.show()

# Plot_3: No of convenience stores vs Price
x = dataset.iloc[1:, 4].values
y = dataset.iloc[1:, -1].values
plt.scatter(x,y)
plt.title('No of convenience stores vs Price')
plt.xlabel('No of convenience stores')
plt.ylabel('Price (per unit area)')
plt.show()

# Plot_4: Longitude vs Price
lo = dataset.iloc[1:, 6].values
y = dataset.iloc[1:, -1].values
plt.scatter(lo,y,color='c',label='Longitude')
plt.title('Longitude vs Price')
plt.xlabel('Longitude')
plt.ylabel('Price (per unit area)')
plt.legend()
plt.show()

# Plot_5: Latitude vs Price
lt = dataset.iloc[1:, 5].values
y = dataset.iloc[1:, -1].values
plt.scatter(lt,y,color='r',label='Latitude')
plt.title('Latitude vs Price')
plt.xlabel('Latitude')
plt.ylabel('Price (per unit area)')
plt.legend()
plt.show()