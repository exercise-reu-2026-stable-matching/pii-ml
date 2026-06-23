import pandas as pd
file = open("stateData_2000_10.csv", "w")
data = open("stateData_100000_10.csv")
df = pd.read_csv(data)
switch = True
for i in range(len(df)):
    value = df.at[i, "converges"]
    if value % 2 == switch:
        df_new = pd.DataFrame(df.iloc[[i]])
        df_new.to_csv(file, mode='a', index=False, header=False)
        switch = not switch
file.close()
data.close()
    
