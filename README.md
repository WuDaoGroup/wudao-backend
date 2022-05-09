# wudao-backend

- Powered by FastAPI, postgresql

1. 上传数据文件 data.csv, username/data.csv
2. 选择预测目标和特征 data.csv_selected_features.csv, username/data_target_confirmed.csv
3. 数据标准化后 data.csv_zscore.csv, username/data_zscore.csv
4. 缺失填充后 data.csv_zscore_fill.csv, username/data_zscore_fill.csv
5. 经过了离群值筛选 data.csv_zscore_fill_filter.csv, username/data_zscore_fill_filter.csv

## Run

```bash
podman pull tualatinx/wudao-backend
podman run -p 8123:8123 -it tualatinx/wudao-backend:latest bash
uvicorn main:app --host=0.0.0.0 --port=${PORT:-8123} --reload --reload-include='*.py'
```
