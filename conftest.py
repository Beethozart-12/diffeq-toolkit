"""pytest 全局配置：保证无图形界面（headless）环境下测试稳定通过。

`diffeq.plotting` 在导入时会加载 matplotlib。若测试收集顺序先于
`test_plotting.py` 设置后端，在无 DISPLAY 的 CI / 服务器上可能触发
"cannot connect to display"。这里在收集任何测试前强制使用 Agg 后端，
消除该环境依赖。
"""

import matplotlib

matplotlib.use("Agg")
