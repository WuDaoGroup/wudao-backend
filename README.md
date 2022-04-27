# wudao-backend

- Powered by FastAPI, postgresql

1. 上传数据文件 data.csv, username/data.csv
2. 选择预测目标和特征 data.csv_selected_features.csv, username/data_target_confirmed.csv
3. 数据标准化后 data.csv_zscore.csv, username/data_zscore.csv
4. 缺失填充后 data.csv_zscore_fill.csv, username/data_zscore_fill.csv
5. 经过了离群值筛选 data.csv_zscore_fill_filter.csv, username/data_zscore_fill_filter.csv

经过修改后，每位用户有一个文件夹，文件夹下的文件名是固定重命名的。

## TODOs

- [ ] notears论文引用
- [ ] 数据预处理最前面加一步 `LabelEncoder()`