# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018.8.9
@Description: 请在有图形界面的机器上运行，否则...
"""
from utils.draw import draw_all_parsed


if __name__ == "__main__":
    """
        本地机器运行draw文件
    """
    user_choice = input("是否在当前机器画图(y/n)：")
    if user_choice == "y":
        try:
            draw_all_parsed()
        except NameError:
            print("当前机器（服务器）不支持画图，请更换到PC机执行！")
