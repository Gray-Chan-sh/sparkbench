# sparkbench — GB10 / DGX Spark 性能压榨套件

目标不是"跑通某个模型"，而是回答一个问题：**这台机器能被压到标称性能的百分之多少，瓶颈在哪。**

五组测量，每组独立可续跑，结果落 `results/results.jsonl`：

| 组 | 测什么 | 关键产出 |
|---|---|---|
| A | BF16/FP16/FP8/NVFP4 矩阵乘 | 实测 TFLOP/s 与标称值的比值 |
| B | 显存带宽（copy/read/write） | 实测 GB/s，并反推"多大的模型能跑多少 tok/s" |
| C | 统一内存可分配上限 | 真正能给 GPU 用多少 GiB，超了是 OOM 还是掉速 |
| D | 视频解码 + 三个 VLM 的帧数阶梯 | 帧数 → prompt token → 延迟/显存曲线 |
| E | 持续满载 30 min | 首窗 vs 末窗，量化降频衰减 |

---

## 1. 三个模型（D 组）

选型逻辑：**同一个家族、三种形态**，这样变量只落在"稠密 vs MoE"和"参数量"上，arm64 上也只有一套 processor 代码路径（少一个失败面，这在 3 小时窗口里很值钱）。

| # | 模型 | 形态 | bf16 权重 | 它回答的问题 |
|---|---|---|---|---|
| 1 | `Qwen/Qwen3-VL-8B-Instruct` | 稠密 8B | ~17 GB | 基线。工具链正常时"应该有多快"，同时给出每帧 token 数和 TTFT 曲线 |
| 2 | `Qwen/Qwen3-VL-30B-A3B-Instruct` | MoE 30B / **激活 3B** | ~61 GB | **核心实验**：273 GB/s 下，MoE 能不能把带宽劣势绕过去 |
| 3 | `Qwen/Qwen3-VL-32B-Instruct` | 稠密 32B | ~65 GB | 带宽墙。稠密大模型的解码被 273 GB/s 直接掐死，是这台机器的下界 |

为什么第 2、3 个是这组对比：**体量几乎一样（30B vs 32B），激活参数差 10 倍**。如果 MoE 的 tok/s 是稠密 32B 的 5–10 倍，那这就是你这次测试最重要的结论——"GB10 适合跑 MoE，不适合跑大稠密模型"。

按 231 GB/s 实测带宽反推的**预期**（用来打勾/打叉，不是结论）：

| 模型 | 每 token 需读 | 解码理论上限 | 现实预期 |
|---|---|---|---|
| 8B dense | ~17 GB | ~13.6 tok/s | 9–12 |
| 30B-A3B MoE | ~6 GB | ~38 tok/s | 20–32 |
| 32B dense | ~65 GB | ~3.6 tok/s | 2.5–3.3 |

时间不够就砍第 3 个：它只是把"下界"钉死，信息量不如前两个。

可选第四个（有富余机时再加）：`Qwen/Qwen2.5-VL-72B-Instruct-AWQ`（~40 GB，测 128 GB 容量边缘 + 稠密墙的极限）。

> 依赖：`pip install "transformers>=4.57"`。Qwen3-VL 走 `AutoModelForImageTextToText`；如果你的版本不认，装 main 分支即可。

---

## 2. 本地先验（不要钱，10 分钟）

```bash
cd sparkbench
python -m sparkbench smoke --profile gb10     # 本机会 FAIL，正常，只是验证脚本能跑
python tools/report.py --profile gb10          # 空数据也能出空表
```

---

## 3. 拿到机器（vast.ai）

```bash
export VAST_API_KEY=...        # console.vast.ai -> Account -> API Keys

python tools/vast.py offers --gpu GB10 --max-price 1.5
python tools/vast.py watch  --gpu GB10 --interval 60     # 有货就通知
```

挑的时候看三件事：**`cpu_arch` 必须是 aarch64**、`inet_down` ≥ 300 Mbps（拉三个模型）、`reliability2` ≥ 0.95。

下单：

```bash
python tools/vast.py create <offer_id> \
  --image <你的镜像> \
  --disk 200 \
  --onstart onstart.sh
```

---

## 4. 构建 arm64 镜像（GitHub Actions，零本地依赖）

本机没有 Docker 也能出镜像。`sparkbench/` 已推到 `github.com/Gray-Chan-sh/sparkbench`，推 `v*` tag 即触发 `.github/workflows/image.yml`。

**已构建通过**：`ghcr.io/gray-chan-sh/sparkbench:latest`（linux/arm64，匿名可拉）。

本机拉取验证：

```bash
docker pull ghcr.io/gray-chan-sh/sparkbench:latest
```

- **不配任何 secret 也能跑**：走公开基座 `nvidia/cuda:13.0.0-devel-ubuntu24.04` + pip 装 cu130 torch（当前 `latest` 就是这么来的）。
- **配了 `NGC_API_KEY`**（repo → Settings → Secrets → Actions）：自动换成 `nvcr.io/nvidia/pytorch:25.09-py3`，sm_121 kernel 更稳。
- 构建期有硬闸门：`torch.cuda.get_arch_list()` 里没有 `sm_100/103/110/120/121` 就直接失败，避免推一个只能 JIT 的废镜像。
- 触发方式：`git tag v0.1.x && git push origin v0.1.x`（推 main 不会触发，这是 `push.tags` 过滤器的预期行为）。

> 注意：GitHub 不允许在 step 的 `if` 里引用 `secrets`（会直接判定 workflow 非法，0 个 job）。本 workflow 里用 shell 判断代替。

---

## 5. 上机后的 3 小时（严格时间盒）

```bash
cd /workspace/sparkbench
bash bootstrap.sh                            # 0:00-0:05
python -m sparkbench smoke --profile gb10    # 0:05-0:15  ← 不过就停，换机器
python -m sparkbench a --profile gb10        # 0:15-0:30  算力
python -m sparkbench b --profile gb10        # 0:30-0:40  带宽
python -m sparkbench c --profile gb10        # 0:40-0:50  容量天花板
python -m sparkbench d --profile gb10        # 0:50-2:30  三个模型 × 帧数阶梯
python -m sparkbench e --profile gb10        # 2:30-3:00  持续负载
python tools/report.py --profile gb10        # 出结论
```

单跑某个模型：`python -m sparkbench d --profile gb10 --model 30b-a3b`

任何一段超时就 `Ctrl-C` 跳过——**已完成的项不会重跑**（除非加 `--no-resume`）。

一键版（自动进 tmux、smoke 门失败即停、跑完出报告 + 审计 + 推送）：

```bash
bash run.sh                    # 全流程
bash run.sh --phases a,b       # 只跑两段
```

---

## 可还原性（和 canguoai 项目同一套标准）

```bash
python -m sparkbench audit                                  # 审计当前 results/
python -m sparkbench audit --results sb-results/latest/results   # 审计拉回本地的快照
```

三档判定：`reproducible` / `replay only` / `cannot replay`，写进 `audit.json`。

运行时自动记录：代码 commit（`env.json` → `git.commit`）、依赖锁（`requirements.lock`）、
权重版本（`revisions.json` + 每条的 `model_revision`）、逐文件哈希（`manifest.json`）、
以及容器/统一内存事实（`env.json` → `memory`）。

**容器场景下最值钱的一条检查**在 smoke 里：如果 cgroup 内存上限低于统一内存池，
它会直接判 FAIL——否则你会在“128GB 的机器”上跑三小时，实际只用到了 64GB。

---

## 播客素材

```bash
python -m sparkbench shots                     # 分镜脚本（8 个镜头，按 A–E 排）
python -m sparkbench.capture run 03-compute -- python -m sparkbench a --profile gb10
```

终端录制走 pty（不是管道），所以 `tqdm`、颜色、`isatty()` 行为都和在真终端里一致。
产物在 `results/captures/*.cast`，asciinema v2 格式，可直接播放或 `agg` 转 GIF。
有 `DISPLAY` 时才启用截图/桌面录屏，没有就如实报 unavailable。

---

## 把结果拉回本地

```bash
python tools/pull.py --host <ip> --port <port> --key ~/.ssh/id_ed25519 --interval 600
```

只读 `tar`，不碰正在写的文件；每 10 分钟几 MB。落地 `<dest>/latest/` + `history/*.tgz`。
`--with-videos` 才会带上桌面录屏这类大文件（mtime 保护，解包后删档）。

---

## 6. 长视频那一段怎么配

D1（解码）不需要模型，把 profile 里 `video.video` 指到样片即可。

D2 的模型在 `profiles/gb10.json` 的 `video.models` 里，加一项就多测一个：

```json
{ "label": "72b-awq", "id": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
  "dtype": "bfloat16", "frames": [16, 64] }
```

它会：均匀抽帧 → 当多图输入 → 测 TTFT / 总耗时 / 峰值显存。
**这是"帧数 × 每帧 token"的粗网格，不是逐秒扫全片**——先找拐点，再决定为哪个点单独长测。

---

## 7. 读数怎么解释

- **A 的比值**：BF16 跑到 60–75% 算正常；标称 1 PFLOP 是稀疏 FP4 折算的，别拿它当基准。
- **B 的绝对值**：这是 GB10 的真实定位。273 GB/s 意味着稠密大模型解码必然难看，MoE 才有活路。
- **C 的 cliff**：统一内存机器通常不会 OOM，而是在某个点突然掉到 swap。那个 GiB 数才是"实际可用显存"。
- **D 的拐点**：帧数翻倍而 TTFT 超线性增长的那一档，就是你实际能用的时长上限。
- **E 的衰减**：140W 的小机器，前 3 分钟和跑半小时不是一个数，报稳态值。

---

## 8. 收工

```bash
python tools/vast.py destroy <instance_id>     # 不停实例会一直计费
```

结果在 `results/results.jsonl`，`smi-*.csv` 是全程时钟/功耗/温度采样。

---

## 附：为什么不用 flash-attn / xformers / decord

PyPI 上这三个包**没有 aarch64 轮子**（flash-attn 2.8.3 / xformers 0.0.35 / decord 0.6.0 的发布文件里 arm64 数量都是 0）。
在 GB10 上装它们 = 源码编译 1–2 小时且大概率失败。本套件全程用 PyTorch 自带 SDPA + PyAV。
