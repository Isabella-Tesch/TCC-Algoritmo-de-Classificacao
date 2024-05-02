import tensorflow as tf
from tensorflow import keras
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QMouseEvent, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import QPoint, Qt, QRect
import numpy as np

class TrainModel():
    def __init__(self):
        # Carregar o conjunto de dados MNIST
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Normaliza os valores dos pixels para o intervalo [0, 1]
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax') #A função de ativação softmax é usada para converter as saídas em probabilidades, assim é possível saber a probabilidade de cada classe para uma determinada entrada.
        ])


        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=5)

        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        print('Acurácia do teste:', test_accuracy)
    
    def PredictNumber(self, image):
        image_resized = tf.image.resize(image, (28, 28))

        #converte imagem para um formato adequado e a normaliza
        image_normalized = np.expand_dims(image_resized, axis=0) // 255.0

        prediction = np.argmax(self.model.predict(image_normalized))
        return prediction


class DrawNumber(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draw Number")
        self.setFixedSize(500, 500)

        #Criar uma label
        # CentralWidget = QWidget()
        # self.setCentralWidget(CentralWidget)
        # layout = QVBoxLayout(CentralWidget)

        # label = QLabel('Olá, Mundo!')
        # layout.addWidget(label)

        self.pixmap = QPixmap(400, 400)
        self.pixmap.fill(Qt.white)

        self.label = QLabel(self)
        self.label.setPixmap(self.pixmap)
        self.label.setGeometry(10, 50, 400, 400)

        self.painter = QPainter(self.pixmap)
        self.painter.setPen(QPen(Qt.black))

        self.squarefill = False

        self.Matriz()

    
    def Matriz(self):
        space = 400 / 28
        x = space
        y = space

        for squarevertical in range(28):
            self.painter.drawLine(int(x), 0, int(x), 400)
            x += space 

        for squarehorizontal in range(28):
            self.painter.drawLine(0, int(y), 400, int(y))
            y += space
        
        self.label.setPixmap(self.pixmap)
    
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            space = 400 / 28
            pos_x = event.pos().x() - 10 # Ajuste para compensar a margem do QLabel
            pos_y = event.pos().y() - 50 # Ajuste para compensar a margem do QLabel

            find_square_x = int(pos_x/space)
            find_square_y = int(pos_y/space)

            
            if self.squarefill: #Se o quadrado estiver preenchido, pinte-o com a cor branca
                 #Preenchimento do quadrado exato onde o mouse foi clicado
                self.painter.fillRect(int(find_square_x * space), int(find_square_y * space), int(space) + 1, int(space) + 1, Qt.white)
                self.squarefill = False
            else:
                self.painter.fillRect(int(find_square_x * space), int(find_square_y * space), int(space) + 1, int(space) + 1, Qt.black)
                self.squarefill = True
            
            #Desenhar as linhas novamente
            self.Matriz()
    
    def getDrawing(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = DrawNumber()

    window.show()

    sys.exit(app.exec_())

    




