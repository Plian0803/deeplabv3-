#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
旧 2_generate_PM.py 的精简包装版
真正的伪标签生成逻辑已迁入  network/psm/generate_psm.py
"""

import argparse
from network.psm.generate_psm import run_generate   # 新模块
from network.psm.generate_psm import parse_cli      # 复用同一套参数


def main():
    args = parse_cli()          # 直接沿用新模块里的参数定义
    run_generate(args)          # 把工作交给新模块


if __name__ == "__main__":
    main()
