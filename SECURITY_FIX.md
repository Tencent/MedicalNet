# MedicalNet 本地安全修复说明

本地修订标识：medicalnet-local-security-20261012。它不是 Tencent 官方发布版本，也不代表上游已合并补丁。

## 结论与边界

1. 原版数据集、推理尺寸恢复及标签评估对列表中的影像路径缺少根目录约束，存在路径越界读取缺陷。实际利用要求不可信列表被更高权限的训练/推理进程消费；普通用户在自己的权限内指定本地文件，不自动构成越权。
2. 原版 train.py/test.py 使用未显式指定 weights_only 的 torch.load。现代 PyTorch 默认会拦截自定义反序列化调用，但旧版默认或人为关闭保护的环境会暴露代码执行风险。当前 requirements.txt 已要求 torch>=2.6，而旧 README 仍写 0.4.1，不能不分运行配置直接宣称当前默认环境 RCE。
3. 项目没有报告中展示的 HTTP API。通用文本读取函数存在，不等于已验证远程任意文本泄露。报告中的外部 Harness 不作为产品自带接口证据。

## 安全修改

- utils/path_security.py：规范化真实路径、要求路径位于可信 data_root 内；拒绝影像绝对路径、盘符相对路径、UNC、父目录跳转、NTFS 流和非法扩展名；仅允许 .nii/.nii.gz；静态符号链接和 Windows junction 指向根外时拒绝。
- utils/file_process.py：root_dir 变成必填参数；列表自身必须是 data_root 内的 .txt；每行一列或两列，逐项校验并忽略空行。训练/带评估的 test.py 要求两列。
- datasets/brains18.py、test.py：所有影像及标签加载点使用相同检查，不再依赖可被 python -O 去掉的文件断言；推理复用已经校验的列表，不重复无约束地打开文件。
- utils/checkpoint.py：三个模型入口统一使用 torch.load(..., weights_only=True, map_location='cpu')；拒绝 PyTorch<2.6；加载时隔离额外 safe globals，完成或异常后恢复；不进行 weights_only=False 降级；验证 state_dict 和恢复训练所需优化器结构。
- model.py：移除错误的 __name__ 重写与字符串 SAFE_GLOBALS，不自动信任自定义模型类。
- requirements.txt / README.md：统一到现代 Python/PyTorch 依赖；移除与其冲突的旧 numpy、SciPy、nibabel 精确版本和过时 argparse/pip 运行依赖。

## 必要兼容调整

现代 nibabel 已移除 get_data()，改用 np.asanyarray(image.dataobj) 保留原读取语义；SciPy 使用 ndimage.zoom。推理将概率 Tensor 显式转换为 CPU NumPy 后再缩放。

恢复训练的测试需要允许禁用预训练时使用普通参数迭代器；保存检查点将原有 ecpoch 拼写修正为 epoch，读取日志兼容旧键。指定不存在的恢复文件现在明确报错，不会悄悄从头训练。未改模型网络结构、损失函数、裁剪逻辑、调度策略或训练目标。

## 安装与验证

使用独立 Python 3.10+ 环境。requirements.txt 的下限用于声明支持策略，并不表示全部依赖组合都已验证。requirements-tested-cpu.txt 保存本次 Windows/Python 3.13 CPU 验证环境的精确版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested-cpu.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -O -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train.py --ci_test
```

GPU 部署请另选与 GPU 驱动匹配的 PyTorch/CUDA 构建，再执行实际业务数据及预训练权重回归。本次未验证 GPU、全量 ResNet 深度或第三方历史检查点。

## 输入迁移

--data_root 必须由可信操作员或平台固定，不能让非信任租户自行选择其他租户目录。

--img_list 保持原有“相对当前工作目录或绝对路径”的解释，不是改为相对 data_root；但其最终真实位置必须在 data_root 内。原来存放在根外的列表必须迁入根内。

```text
data/
  train.txt
  images/case.nii.gz
  labels/case.nii.gz
```

train.txt 示例：

```text
images/case.nii.gz labels/case.nii.gz
```

路径按空白分列，与原项目一致，不支持文件名含空格。合法 checkpoint 必须为包含 state_dict 的字典，值为 Tensor；恢复训练还需要 optimizer。序列化整个模型对象或自定义对象的旧文件将被拒绝。不要为兼容它们关闭安全模式或照异常提示扩张白名单。

## 部署防护与残余风险

- 只从可信渠道获得检查点，并用可信渠道提供的哈希或签名核验来源；本补丁包的哈希不是模型文件的可信摘要。项目未提供可信模型摘要，因此没有硬编码虚构的模型 SHA-256。
- weights_only 降低不可信 pickle 的执行面，不是恶意文件的完整安全沙箱，也不解决内存/CPU 耗尽等依赖层风险。仍需及时升级依赖、限制资源并以低权限运行。
- 路径检查不是操作系统沙箱。数据目录及父目录不可由攻击者并发改写；检查与打开之间的 TOCTOU、攻击者预置硬链接以及任意挂载权限不在本补丁保证范围。多租户部署应只读挂载、容器隔离与最小文件权限配合。
- checkpoint 的额外白名单注册表在加载时临时隔离。各加载调用在本模块内串行；外部线程不能同时注册 globals 或绕过本入口调用 torch.load。每个训练作业建议运行在独立进程。
- 本次是针对两份报告的定向验证，不是全仓所有类别漏洞审计，也没有对报告中的远程 API、正式公开状态或全部上游历史版本作背书。
