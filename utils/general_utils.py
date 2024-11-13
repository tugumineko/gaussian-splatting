import random
import sys
from datetime import datetime

import numpy as np
import torch


# 重定义输出流，定义输出末尾是有’\n‘时,则输出带时间戳
# 若设置为静默模式，则print函数无效，不会输出任何内容
def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self,silent):
            self.silent = silent

        def write(self,x):
            if not self.silent:
                if x.endwith("\n"):
                    old_f.write(x.replace("\n", "[{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    # 设定随机数种子，使不同人的实验结果一致
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0")) # 指定cuda使用第一个GPU
