# Harness Engineering Dataset

## 数据来源

本项目包含来自多个来源的数据集：

1. **SII-HE Dataset**: https://github.com/Chicken5674/sii-he-dataset
2. **2026 创智学院暑期实训数据集**: https://github.com/edgerunneres/2026-chuangzhi-academy-summer-camp-harness-engineering-mock-dataset
3. **百度网盘**: https://pan.baidu.com/s/1Yv3NOAKTwe8Ax97FZFdGuQ  
   提取码: `7i22`
4. **部分自建数据**

---

## 文件夹说明与运行方法

### data1 - 多标签分类数据集

包含多种类型的测试和训练数据（dev, mcq, ood 等），支持多语言（英文和中文）。

**运行所有任务**：
```bash
# 方法1：逐个运行每种数据类型
conda run --no-capture-output -n HE python -u run.py \
  --train data1/train_dev.jsonl \
  --dev data1/test_dev.jsonl \
  --workers 20 \
  --runs 1

conda run --no-capture-output -n HE python -u run.py \
  --train data1/train_mcq.jsonl \
  --dev data1/test_mcq.jsonl \
  --workers 20 \
  --runs 1

conda run --no-capture-output -n HE python -u run.py \
  --train data1/train_ood.jsonl \
  --dev data1/test_ood.jsonl \
  --workers 20 \
  --runs 1

# 方法2：批量运行脚本
for variant in dev mcq ood; do
  echo "===== Running $variant ====="
  conda run --no-capture-output -n HE python -u run.py \
    --train "data1/train_${variant}.jsonl" \
    --dev "data1/test_${variant}.jsonl" \
    --workers 20 \
    --runs 1
done
```

---

### data2 - 多任务学习数据集

包含 5 个不同的任务：task1、task2_education、task2_restaurant、task2_techsupport、task3

**运行所有任务**：
```bash
for d in data2/*; do
  if [ -f "$d/train.jsonl" ] && [ -f "$d/test.jsonl" ]; then
    echo "===== Running $d ====="
    conda run --no-capture-output -n HE python -u run.py \
      --train "$d/train.jsonl" \
      --dev "$d/test.jsonl" \
      --workers 20 \
      --runs 1
  fi
done
```

---

### data3 - 多任务与多域评估数据集

包含 6 个领域分割（fin、hr、log、pol、sec、sw）和 4 个综合任务

**运行多任务评估**：
```bash
conda run -n HE python data3/multi_task_eval.py \
  --student-run run.py \
  --workers 20 \
  --runs 1
```

**数据验证**（可选）：
```bash
conda run -n HE python data3/validate_dataset.py
```

---

## 环境设置

所有命令使用 conda 环境 `HE`。确保已创建该环境：

```bash
conda create -n HE python=3.13  # 或其他版本
conda activate HE
pip install -r requirements.txt  # 如果有的话
```

---

## 注意事项

- 确保 `run.py` 文件在项目根目录
- `--workers` 参数可根据您的硬件调整
- `--runs` 参数指定运行次数
- 运行可能耗时较长，建议在后台运行或使用 `nohup`
