from pathlib import Path
import pandas as pd

frame = pd.read_csv(Path('douro_vineyard_data.csv'))
frame['distance_group'] = pd.cut(frame['distance_from_porto_km'],
    bins=[float('-inf'), 104, 159.4, float('inf')],
    labels=['<=104', '(104,159.4]', '>159.4'])
print('Overall:')
print(frame.groupby('treatment')['harvest_success'].agg(['mean', 'size']))
print('\nBy distance group:')
print(frame.groupby(['distance_group', 'treatment'], observed=True)['harvest_success'].agg(['mean', 'size']))
