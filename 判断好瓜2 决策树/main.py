import pandas as pd
import my_functions as fc


data = pd.read_excel('raw_data.xlsx')
continuous_feature = {"密度", "含糖率"}

my_tree = fc.GenerateTree(data, "好瓜", ["密度", "含糖率"])
