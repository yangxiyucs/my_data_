import pandas as pd

data = {'name': ['tom', 'krru', 'bun'], 'class': [1, 2, 3], 'sex': ['girl', 'boy', 'boy']}

df = pd.DataFrame(data=data)
path = 'text.xls'
df.to_excel(excel_writer=path,sheet_name='xx')

