import pandas as pd



data = pd.read_csv("data1.csv")

print(eval(data["Decoder"].values[0]))

print(eval(data["Encoder"].values[0]))