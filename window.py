import wx
import os
import torch
import image
import pandas as pd


class Window(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(
            self, None, title="Hand-Written Digits Recognizor", size=(700, 700)
        )
        self.panel = wx.Panel(self, -1)
        self.select = wx.Button(self.panel, label=u"选择", pos=(305, 5), size=(80, 25))
        self.select.Bind(wx.EVT_BUTTON, self.on_open_file)
        self.recognize = wx.Button(self.panel, label=u"识别", pos=(405, 5), size=(80, 25))
        self.recognize.Bind(wx.EVT_BUTTON, self.get_digit)
        self.file_name = wx.TextCtrl(self.panel, pos=(5, 5), size=(230, 25))
        self.model = torch.load("./model.pkl")

    def on_open_file(self, event):
        wildcard = "All files(*.*)|*.*"
        dialog = wx.FileDialog(None, "select", os.getcwd(), "", wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            self.file_name.SetValue(dialog.GetPath())
            dialog.Destroy
        self.image_wx = wx.Image(self.file_name.GetValue(), wx.BITMAP_TYPE_JPEG)
        self.image_pil = image.get_image(self.file_name.GetValue())
        self.show()

    def get_digit(self, event):
        pred = image.predict(model=self.model, image=self.image_pil)
        self.answer = wx.StaticText(self.panel, -1, str(pred.idxmax()[0]), (600, 10))

    def show(self):
        temp = self.image_wx.ConvertToBitmap()
        size = temp.GetWidth(), temp.GetHeight()
        self.bmp = wx.StaticBitmap(self.panel, -1, temp, pos=(50, 50), size=size)
        self.bmp.SetBitmap(temp)

