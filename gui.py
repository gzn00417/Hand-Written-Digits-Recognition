# import wx
# from window import Window

# app = wx.App()
# SiteFrame = Window()
# SiteFrame.Show()
# app.MainLoop()

import sys
from PyQt5.QtWidgets import QApplication
from written_board import WrittenBoard

app = QApplication(sys.argv)
board = WrittenBoard()  # 新建一个主界面
board.show()  # 显示主界面
exit(app.exec_())  # 进入消息循环
