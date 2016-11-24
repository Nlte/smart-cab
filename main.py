import pandas as pd

df_list = []
with open('./gridsearch_results.txt') as f:
    for line in f:
        # remove whitespace at the start and the newline at the end
        line = line.strip()
        # split each column on whitespace
        c = line.split()
        col = [c[1], c[3], c[5], c[7], c[9]]
        df_list.append(col)
print df_list
df = pd.DataFrame(df_list, columns=['epsilon', 'alpha', 'gamma', 'avgresult', 'avgresult_p'])
print df

df.to_csv('gridsearch_results.csv')
