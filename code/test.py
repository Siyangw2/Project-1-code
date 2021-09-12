import folium
import pandas as pd
#
# data = pd.read_csv('data/green_tripdata_2020-04.csv')
# print(data.columns)
aa = folium.Map(location=[27.664827, -81.516], zoom_start = 7, tiles='Stamen Toner')
aa.save('a.html')