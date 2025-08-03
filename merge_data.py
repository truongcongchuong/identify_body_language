import pandas as pd

# # Đọc hai tệp CSV
# df1 = pd.read_csv('../data/label_data.csv', header=None, encoding='latin1')
# df2 = pd.read_csv('../data1/label_data.csv', header=None, encoding='latin1')

# # Gộp hai DataFrame lại với nhau theo chiều dọc
# df = pd.concat([df1, df2], axis=0)

# # Lưu DataFrame gộp vào một tệp CSV mới
# df.to_csv('data/label_data.csv', index=False)

# # Đọc hai tệp CSV
# df1 = pd.read_csv('../data/training_data.csv', header=None, encoding='latin1')
# df2 = pd.read_csv('../data1/training_data.csv', header=None, encoding='latin1')

# # Gộp hai DataFrame lại với nhau theo chiều dọc
# df = pd.concat([df1, df2], axis=0)

# # Lưu DataFrame gộp vào một tệp CSV mới
# df.to_csv('data/training_data.csv', index=False)


path_train = "data/training_data.csv"
path_label = "data/label_data.csv"
# df_feature = pd.read_csv(path_train, header=None, encoding='latin1')
df_label = pd.read_csv(path_label, header=None)

print(df_label.head())
print(pd.__version__)