from PyQt5.Qt import (
    QColor,
    QIcon,
    QMessageBox,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
    QPoint,
    QSize,
)
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSplitter,
    QComboBox,
    QLabel,
    QSpinBox,
    QFileDialog,
    QCheckBox,
)
from PyQt5.QtCore import Qt
from torch import load


import image


class PaintBoard(QWidget):
    def __init__(self, Parent=None):
        """
        Constructor
        """
        super().__init__(Parent)

        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()

    def __InitData(self):

        self.__size = QSize(480, 460)

        # 新建QPixmap作为画板，尺寸为__size
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.white)  # 用白色填充画板

        self.__IsEmpty = True  # 默认为空画板
        self.EraserMode = False  # 默认为禁用橡皮擦模式

        self.__lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.__painter = QPainter()  # 新建绘图工具

        self.__thickness = 10  # 默认画笔粗细为10px
        self.__penColor = QColor("black")  # 设置默认画笔颜色为黑色
        self.__colorList = QColor.colorNames()  # 获取颜色列表

    def __InitView(self):
        # 设置界面的尺寸为__size
        self.setFixedSize(self.__size)

    def Clear(self):
        # 清空画板
        self.__board.fill(Qt.white)
        self.update()
        self.__IsEmpty = True

    def ChangePenColor(self, color="black"):
        # 改变画笔颜色
        self.__penColor = QColor(color)

    def ChangePenThickness(self, thickness=10):
        # 改变画笔粗细
        self.__thickness = thickness

    def IsEmpty(self):
        # 返回画板是否为空
        return self.__IsEmpty

    def GetContentAsQImage(self):
        # 获取画板内容（返回QImage）
        image = self.__board.toImage()
        return image

    def paintEvent(self, paintEvent):
        # 绘图事件
        # 绘图时必须使用QPainter的实例，此处为__painter
        # 绘图在begin()函数与end()函数间进行
        # begin(param)的参数要指定绘图设备，即把图画在哪里
        # drawPixmap用于绘制QPixmap类型的对象
        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.__board)

        if self.EraserMode == False:
            # 非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))  # 设置画笔颜色，粗细
        else:
            # 橡皮擦模式下画笔为纯白色，粗细为10
            self.__painter.setPen(QPen(Qt.white, 10))

        # 画线
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()  # 更新显示

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False  # 画板不再为空


class WrittenBoard(QWidget):
    def __init__(self, Parent=None):
        """
        Constructor
        """
        super().__init__(Parent)
        self.model = load("./model.pkl")
        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()

    def __InitData(self):
        """
                  初始化成员变量
        """
        self.__paintBoard = PaintBoard(self)
        # 获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        """
                  初始化界面
        """
        self.setFixedSize(640, 480)
        self.setWindowTitle("手写数字识别手写板")

        self.label_name = QLabel("郭茁宁x许健", self)
        self.label_name.setGeometry(500, 5, 120, 35)

        self.label_name = QLabel(u"Hand-Written", self)
        self.label_name.setGeometry(500, 35, 100, 35)

        self.label_name = QLabel("Digits", self)
        self.label_name.setGeometry(500, 65, 100, 35)

        self.label_name = QLabel("Recognizor", self)
        self.label_name.setGeometry(500, 95, 100, 35)

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为5px
        sub_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__btn_Recognize = QPushButton("开始识别")
        self.__btn_Recognize.setParent(self)
        self.__btn_Recognize.clicked.connect(self.on_btn_Recognize_Clicked)
        sub_layout.addWidget(self.__btn_Recognize)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面

        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("保存")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__cbtn_Eraser = QCheckBox("  使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(30)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(10)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange
        )  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)  # 用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(
            self.on_PenColorChange
        )  # 关联下拉列表的当前索引变更信号与函数on_PenColorChange
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)  # 将子布局加入主布局

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        save_path = QFileDialog.getSaveFileName(self, "Save Your Paint", ".\\", "*.png")
        print(save_path)
        if save_path[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(save_path[0])
        print(save_path[0])

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式

    def on_btn_Recognize_Clicked(self):
        save_path = "./written_digit.jpg"
        self.__paintBoard.GetContentAsQImage().save(save_path)
        ans, pred = image.predict(model=self.model, image=image.get_image(save_path))
        QMessageBox.question(self, u"识别结果", u"手写数字识别为：" + str(ans), QMessageBox.Yes)

    def Quit(self):
        self.close()

