# 库存约束下的在线 Top-K 推荐

课程项目 *Online Top-K Recommendation* 的提交材料：Python / `gurobipy` 代码、全部实验结果与 LaTeX 论文。

首页按作者要求暂留组员署名栏。正式提交前，可在 `paper/main.tex` 的 `\author{...}` 中填写姓名并重新编译。

## 目录

```
online_recommendation_report.pdf   论文（与 paper/main.pdf 相同）
修改摘要.pdf                  本次修订的内容、结果与验证范围
online recommendation/      题目原文与五份到达数据（原样保留）
code/
  recommendation.py         模型、流体 LP、四种在线策略、模拟器、动态规划、精确评价
  run_policy.py             对任意到达 CSV 运行策略，输出逐期推荐、收入、剩余库存与遗憾
  experiments.py            论文中的全部实验（固定随机种子）
  make_assets.py            由 results/ 生成论文的表格、图形与数值宏
  embed_cjk_map.py          嵌入中文字符映射，使 PDF 正文可搜索、复制
  tests.py                  17 项自动化测试
  instance.json             实例参数示例（可改为任意 m, n, T, K）
  requirements.txt
results/                    实验结果（CSV / JSON）
paper/
  main.tex                  论文主文件（XeLaTeX + ctex）
  revision_summary.tex      修改摘要的可编辑源文件
  sections/  tables/  figures/  generated/numbers.tex
  main.pdf                  编译好的论文
```

## 运行

原实验使用 Python 3.11，本次验证使用 conda base 中的 Python 3.12。复现原论文的对偶选择请使用 Gurobi 13.0.3，所需规模在受限许可证范围内。生成中文图表需要 Matplotlib 能识别的中文字体，例如 Microsoft YaHei、Hiragino Sans GB 或 Noto Sans CJK SC。在 `code` 目录下运行：

```bash
python -m pip install -r requirements.txt
python tests.py                                                    # 约 10 秒
python run_policy.py "../online recommendation/data/arrivals_dataset_1.csv"
python run_policy.py my_arrivals.csv --reps 1000                   # 1000 次购买重复取平均
python run_policy.py my_arrivals.csv --instance instance.json      # 其他实例
```

输入 CSV 的列为 `t,i_t`，`t` 从 1 连续编号，类型取 1..m。`--policy` 可选 `LP-Dual`（主策略，默认）、`LP-FD`、`Greedy`、`Alg1`。

在线调用：创建一次策略对象，每期调用 `policy.recommend(i, c, H)`，其中 `i` 为顾客类型（从 0 开始）、`c` 为当前库存、`H` 为含本期在内的剩余期数；观察到购买后由调用方更新库存；结束时 `policy.close()`。

重现论文：

```bash
python experiments.py all     # 16 核约 1 小时；也可单独运行 exact / simulate / count / loss / ...
python make_assets.py
cd ../paper
mkdir -p build
xelatex -output-directory=build main.tex
xelatex -output-directory=build main.tex
xelatex -output-directory=build main.tex
python ../code/embed_cjk_map.py build/main.pdf main.pdf
cp main.pdf ../online_recommendation_report.pdf
xelatex -output-directory=build revision_summary.tex
python ../code/embed_cjk_map.py build/revision_summary.pdf ../修改摘要.pdf
```

`experiments.py` 会在系统临时目录下缓存 LP 动作表（约 100 MB/实例），可随时删除。

## 主要结果（五份数据，精确期望值）

| 策略 | 平均期望收入 | 平均遗憾 | 最大遗憾 |
|---|---|---|---|
| 算法 1（不检查库存） | 693.394 | 12.606 | 12.606 |
| 库存感知贪心 | 695.550 | 10.450 | 11.168 |
| LP 有限差分 | 695.490 | 10.510 | 11.058 |
| **LP 对偶价格（主策略）** | **695.894** | **10.106** | **10.740** |
| 预知类型序列的最优值 | 696.062 | 9.938 | 10.525 |

离线 LP 上界为 706。五份数据上，预知顾客顺序时最高期望收入平均为 696.062，和流体上界相差 9.938。主策略相对库存感知贪心平均多赚 0.344，与预知序列最优策略的差距缩小 67.1%。

按各序列第一次购买模拟的实现值计算，主策略的五条收入为 678、710、680、660、656，平均样本遗憾 29.2，最大样本遗憾 50。表中遗憾对所有购买路径取条件期望；两组数值对应不同的评价口径。
