import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QMessageBox, QGridLayout, QApplication, QPushButton, QLabel, QFileDialog)
from predict import predict
from PIL import Image


class FruitsClassifierWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QGridLayout(self)
        self.label_image = QLabel(self)
        self.label_predict_result = QLabel('识别结果', self)
        self.label_predict_result_display = QLabel(self)

        self.button_search_image = QPushButton('选择图片', self)
        self.button_run = QPushButton('开始识别', self)
        self.setLayout(self.layout)
        self.init_ui()

    # init layout
    def init_ui(self):
        self.layout.addWidget(self.label_image, 1, 1, 3, 2)
        self.layout.addWidget(self.button_search_image, 1, 3, 1, 2)
        self.layout.addWidget(self.button_run, 3, 3, 1, 2)
        self.layout.addWidget(self.label_predict_result, 4, 3, 1, 1)
        self.layout.addWidget(self.label_predict_result_display, 4, 4, 1, 1)

        self.button_search_image.clicked.connect(self.choose_image)
        self.button_run.clicked.connect(self.run)

        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('水果识别')
        self.show()

    # choose image
    def choose_image(self):
        global fname
        img_name, img_type = QFileDialog.getOpenFileName(self, "选择图片", "", "*.jpg;;*.jpeg;;All Files(*)")
        jpg = QPixmap(img_name).scaled(self.label_image.width(), self.label_image.height())
        self.label_image.setPixmap(jpg)
        fname = img_name

    # popup prompt
    def error_prompt(self):
        QMessageBox.warning(self, "错误提示", "识别失败或该水果当前不受支持")

    # predict
    def run(self):
        global fname
        file_name = str(fname)
        img = Image.open(file_name)
        img = img.resize((100, 100))
        try:
            result = predict(img)
            self.label_predict_result_display.setText(result)
        except:
            self.error_prompt()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = FruitsClassifierWindow()
    mainWindow.show()
    sys.exit(app.exec_())
