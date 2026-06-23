import pandas as pd
write_file = open("stateData_2000_10_test.csv", "w")
data_file = open("stateData_100000_10_test.csv")
df = pd.read_csv(data_file)
switch = True
for i in range(len(df)):
    value = df.at[i, "converges"]
    if value % 2 == switch:
        df_new = pd.DataFrame(df.iloc[[i]])
        df_new.to_csv(write_file, mode='a', index=False, header=False)
        switch = not switch
write_file.close()
data_file.close()
