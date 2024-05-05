import pandas as pd
import sys
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QFileDialog, QMdiArea, QMdiSubWindow, QSlider, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QTransform
from PyQt5.QtCore import Qt
from skimage.measure import label, regionprops
from scipy import ndimage




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dijital Görüntü İşleme")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QMdiArea()
        self.setCentralWidget(self.central_widget)

        self.create_menu()
        self.create_main_page()

    def create_menu(self):
        menubar = self.menuBar()
    
        assignment1_menu = menubar.addMenu("Ödev 1: Temel İşlevselliği Oluştur")
        assignment2_menu = menubar.addMenu("Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon")
        assignment3_menu = menubar.addMenu("Ödev 3")
    
        assignment1_action = QAction("Resim Ekle", self)
        assignment1_action.triggered.connect(self.assignment1_clicked)
        assignment1_menu.addAction(assignment1_action)
    
        assignment2_action = QAction("Görüntü Operasyonları ve İnterpolasyon", self)
        assignment2_action.triggered.connect(self.assignment2_clicked)
        assignment2_menu.addAction(assignment2_action)
    
        hough_transform_action = QAction("Çizgi Takip", self)
        hough_transform_action.triggered.connect(self.cizgi_takip)
        assignment3_menu.addAction(hough_transform_action)
    
        assignment3_action = QAction("Sigmoid İşlemleri", self)
        assignment3_action.triggered.connect(self.assignment3_clicked)
        assignment3_menu.addAction(assignment3_action)
        
        assignment4_action = QAction("Göz Tespiti", self)
        assignment4_action.triggered.connect(self.goz_tespiti)
        assignment3_menu.addAction(assignment4_action)
        
        assignment5_action = QAction("Deblurring", self)
        assignment5_action.triggered.connect(self.deblur_image)
        assignment3_menu.addAction(assignment5_action)
        
        assignment6_action = QAction(" Resimdeki nesneleri sayma ve özellik çıkarma", self)
        assignment6_action.triggered.connect(self.koordinat_hesaplama)
        assignment3_menu.addAction(assignment6_action)
        
        
        
    


    
    def cizgi_takip(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Resim Ekle", "", "Resim Dosyaları (*.jpg *.png *.webp)")
        if file_path:
            sub_window = QMdiSubWindow()
            image_label = QLabel()
            pixmap = QPixmap(file_path)
            image_label.setPixmap(pixmap.scaledToWidth(400))
            sub_window.setWidget(image_label)
            sub_window.setWindowTitle("Hough Transform")

            # Orijinal resmi yedekle
            original_pixmap = QPixmap(pixmap)

            # Hough Transform kullanarak çizgileri tespit etme
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

            # Eğer çizgiler varsa, bunları orijinal resme uygula
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # İşaretlenmiş resmi göster
                cv2.imwrite('output.jpg', image)
                result_pixmap = QPixmap('output.jpg')
                result_label = QLabel()
                result_label.setPixmap(result_pixmap.scaledToWidth(400))

                vbox = QVBoxLayout()
                vbox.addWidget(result_label)

                widget = QWidget()
                widget.setLayout(vbox)
                sub_window.setWidget(widget)

                self.central_widget.addSubWindow(sub_window)
                sub_window.show()

            else:
                # Eğer çizgi bulunamazsa, orijinal resmi göster
                image_label.setPixmap(original_pixmap.scaledToWidth(400))
                QMessageBox.warning(self, "Uyarı", "Çizgi bulunamadı!")

    def create_main_page(self):
        title_label = QLabel("Dijital Görüntü İşleme Uygulaması", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setGeometry(0, 50, 800, 50)  
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")  
        title_label.show()

        student_info_label = QLabel("Feyza Ülkü Öztaşkın 211229028", self)
        student_info_label.setAlignment(Qt.AlignCenter)
        student_info_label.setGeometry(0, 550, 800, 50) 
        student_info_label.setStyleSheet("font-size: 12px;")  
        student_info_label.show()

    def assignment1_clicked(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Resim Ekle", "", "Resim Dosyaları (*.jpg *.png)")
        if file_path:
            sub_window = QMdiSubWindow()
            image_label = QLabel()
            pixmap = QPixmap(file_path)
            image_label.setPixmap(pixmap.scaledToWidth(400))
            sub_window.setWidget(image_label)
            sub_window.setWindowTitle("Eklenen Resim")
    
            # Hough Transform kullanarak çizgileri tespit etme
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite('output.jpg', image)
    
            result_pixmap = QPixmap('output.jpg')
            result_label = QLabel()
            result_label.setPixmap(result_pixmap.scaledToWidth(400))
    
            vbox = QVBoxLayout()
            vbox.addWidget(result_label)
    
            widget = QWidget()
            widget.setLayout(vbox)
            sub_window.setWidget(widget)
    
            self.central_widget.addSubWindow(sub_window)
            sub_window.show()


    def assignment2_clicked(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Resim Ekle", "", "Resim Dosyaları (*.jpg *.png)")
        if file_path:
            sub_window = QMdiSubWindow()
            image_label = QLabel()
            pixmap = QPixmap(file_path)
            image_label.setPixmap(pixmap.scaledToWidth(400))
            sub_window.setWidget(image_label)
            sub_window.setWindowTitle("Görüntü Operasyonları ve İnterpolasyon")

            factor_label = QLabel("Büyütme/Küçültme Oranı:", self)
            factor_edit = QLineEdit(self)
            factor_edit.setPlaceholderText("Büyütme/Küçültme Oranını Girin")
            factor_label.setAlignment(Qt.AlignCenter)

            zoom_factor_label = QLabel("Zoom In/Out Oranı:", self)
            zoom_factor_edit = QLineEdit(self)
            zoom_factor_edit.setPlaceholderText("Zoom In/Out Oranını Girin")
            zoom_factor_label.setAlignment(Qt.AlignCenter)

            rotation_label = QLabel("Döndürme Açısı:", self)
            rotation_edit = QLineEdit(self)
            rotation_edit.setPlaceholderText("Döndürme Açısını Girin")
            rotation_label.setAlignment(Qt.AlignCenter)

            enlarge_button = QPushButton("Büyüt", self)
            shrink_button = QPushButton("Küçült", self)
            zoom_in_button = QPushButton("Zoom In", self)
            zoom_out_button = QPushButton("Zoom Out", self)
            rotate_left_button = QPushButton("Sola Döndür", self)
            rotate_right_button = QPushButton("Sağa Döndür", self)

            enlarge_button.clicked.connect(lambda: self.resize_image(image_label, pixmap, factor_edit.text(), True, interpolation="bilinear"))
            shrink_button.clicked.connect(lambda: self.resize_image(image_label, pixmap, factor_edit.text(), False, interpolation="bilinear"))
            zoom_in_button.clicked.connect(lambda: self.zoom_image(image_label, pixmap, zoom_factor_edit.text(), True, interpolation="bicubic"))
            zoom_out_button.clicked.connect(lambda: self.zoom_image(image_label, pixmap, zoom_factor_edit.text(), False, interpolation="bicubic"))
            rotate_left_button.clicked.connect(lambda: self.rotate_image(image_label, pixmap, -float(rotation_edit.text()), pixmap.width(), pixmap.height(), interpolation="average"))
            rotate_right_button.clicked.connect(lambda: self.rotate_image(image_label, pixmap, float(rotation_edit.text()), pixmap.width(), pixmap.height(), interpolation="average"))

            hbox1 = QHBoxLayout()
            hbox1.addWidget(enlarge_button)
            hbox1.addWidget(shrink_button)

            hbox2 = QHBoxLayout()
            hbox2.addWidget(factor_label)
            hbox2.addWidget(factor_edit)

            hbox3 = QHBoxLayout()
            hbox3.addWidget(zoom_in_button)
            hbox3.addWidget(zoom_out_button)

            hbox4 = QHBoxLayout()
            hbox4.addWidget(zoom_factor_label)
            hbox4.addWidget(zoom_factor_edit)

            hbox5 = QHBoxLayout()
            hbox5.addWidget(rotation_label)
            hbox5.addWidget(rotation_edit)

            hbox6 = QHBoxLayout()
            hbox6.addWidget(rotate_left_button)
            hbox6.addWidget(rotate_right_button)

            vbox = QVBoxLayout()
            vbox.addWidget(image_label)
            vbox.addLayout(hbox1)
            vbox.addLayout(hbox2)
            vbox.addLayout(hbox3)
            vbox.addLayout(hbox4)
            vbox.addLayout(hbox5)
            vbox.addLayout(hbox6)

            widget = QWidget()
            widget.setLayout(vbox)
            sub_window.setWidget(widget)

            self.central_widget.addSubWindow(sub_window)
            sub_window.show()


            
            
    def koordinat_hesaplama(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Resim Ekle", "", "Resim Dosyaları (*.jpg *.png)")
        if file_path:
            image = cv2.imread(file_path)
            
            b, g, r = cv2.split(image)
            green_channel = g
    
            _, binary = cv2.threshold(green_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            labeled_image = label(binary)
            regions = regionprops(labeled_image, intensity_image=binary)
    
            df = pd.DataFrame(columns=['No', 'Center', 'Length', 'Width', 'Diagonal', 'Energy', 'Entropy', 'Mean', 'Median'])
    
            for idx, region in enumerate(regions):
                center = region.centroid
                length = region.bbox[2] - region.bbox[0]
                width = region.bbox[3] - region.bbox[1]
                diagonal = np.sqrt(length**2 + width**2)
                energy = np.sum(region.intensity_image ** 2)
                entropy = -np.sum(region.intensity_image * np.log(region.intensity_image + 1e-15))
                mean_intensity = np.mean(region.intensity_image)
                median_intensity = np.median(region.intensity_image)
    
                df.loc[idx] = [idx+1, center, length, width, diagonal, energy, entropy, mean_intensity, median_intensity]
    
            excel_path, _ = QFileDialog.getSaveFileName(self, "Excel Dosyası Kaydet", "", "Excel Dosyaları (*.xlsx)")
            if excel_path:
                df.to_excel(excel_path, index=False)

            
        
            
    def deblur_image(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Resim Ekle", "", "Resim Dosyaları (*.jpg *.png *.jpeg)")
        if file_path:
            original_image = cv2.imread(file_path)

            blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    
            unsharp_image = cv2.addWeighted(original_image, 1.5, blurred_image, -0.5, 0)
    
            combined_image = np.hstack((original_image, unsharp_image))
    
            cv2.imshow("Original vs Deblurred", combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    def fade_image(self, image_label, pixmap, value):
        new_pixmap = QPixmap(pixmap)
        painter = QPainter(new_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        color = QColor(0, 0, 0, int(255 * (1 - (value / 100))))
        painter.fillRect(new_pixmap.rect(), color)
        painter.end()
        image_label.setPixmap(new_pixmap)
        
   
                      

    def resize_image(self, image_label, pixmap, factor, enlarge=True, interpolation="bilinear"):
        try:
            factor = float(factor)
            if factor <= 0:
                raise ValueError
            
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            if enlarge:
                new_width = int(width * factor)
                new_height = int(height * factor)
            else:
                new_width = int(width / factor)
                new_height = int(height / factor)
            
            new_pixmap = pixmap.scaled(new_width, new_height, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            image_label.setPixmap(new_pixmap)

        except ValueError:
            QMessageBox.warning(self, "Uyarı", "Geçerli bir oran girin!")

    def rotate_image(self, image_label, pixmap, angle, width, height, interpolation="average"):
        try:
            transform = QTransform().rotate(angle)
            new_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
            image_label.setPixmap(new_pixmap)
            image_label.setFixedSize(new_pixmap.width(), new_pixmap.height())
        except ValueError:
            QMessageBox.warning(self, "Uyarı", "Geçerli bir açı girin!")

    def zoom_image(self, image_label, pixmap, factor, is_zoom_in, interpolation="bicubic"):
        try:
            factor = float(factor)
            if factor <= 0:
                raise ValueError

            if is_zoom_in:
                new_width = int(pixmap.width() * factor)
                new_height = int(pixmap.height() * factor)
            else:
                new_width = int(pixmap.width() / factor)
                new_height = int(pixmap.height() / factor)

            new_pixmap = pixmap.scaled(new_width, new_height, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            image_label.setPixmap(new_pixmap)
            image_label.setFixedSize(new_pixmap.width(), new_pixmap.height())
        except ValueError:
            QMessageBox.warning(self, "Uyarı", "Geçerli bir oran girin!")
            
    def assignment3_clicked(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Resim Ekle", "", "Resim Dosyaları (*.jpg *.png)")
    
        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
            enhanced_images = [
                self.standard_sigmoid(image),
                self.shifted_sigmoid(image),
                self.sloped_sigmoid(image),
                self.custom_function(image)
            ]
    
            layout = QHBoxLayout()
    
            for enhanced_image in enhanced_images:
                image_label = QLabel()
                pixmap = self.convert_numpy_to_qpixmap(enhanced_image)
                image_label.setPixmap(pixmap)
                layout.addWidget(image_label)
    
            widget = QWidget() 
            widget.setLayout(layout) 
    
            sub_window = QMdiSubWindow()  
            sub_window.setWidget(widget)  
            sub_window.setWindowTitle("Sigmoid İşlemleri")  
    
            self.central_widget.addSubWindow(sub_window) 
            sub_window.show() 

    def standard_sigmoid(self, image):
        return 1 / (1 + np.exp(-image))

    def shifted_sigmoid(self, image):
        return 1 / (1 + np.exp(-(image - 128) / 10))

    def sloped_sigmoid(self, image):
        return 1 / (1 + np.exp(-0.1 * (image - 128)))

    def custom_function(self, image):
        return image

    def standard_sigmoid_contrast(self, image):
        return self.s_curve_contrast(self.standard_sigmoid(image))

    def shifted_sigmoid_contrast(self, image):
        return self.s_curve_contrast(self.shifted_sigmoid(image))

    def sloped_sigmoid_contrast(self, image):
        return self.s_curve_contrast(self.sloped_sigmoid(image))

    def custom_function_contrast(self, image):
        return self.s_curve_contrast(self.custom_function(image))

    def s_curve_contrast(self, image):
        # S-eğrisi eğrisini oluştur
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = np.uint8(255 * (1 / (1 + np.exp(-15 * (i / 255 - 0.5)))))

        # Eğriyi görüntüye uygula
        enhanced_image = cv2.LUT(image, lut)
        return enhanced_image

    def convert_numpy_to_qpixmap(self, image):
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)
            
    def goz_tespiti(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Resim Ekle", "", "Resim Dosyaları (*.jpg *.png *.jpeg)")

    
        if file_path:
         
            resim = cv2.imread(file_path)
            
         
            gozler = self.goz_tespit_hough(resim)
            
        
            self.isaretle_ve_goster(resim, gozler)
    
    def goz_tespit_hough(self, resim):
        gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        
        daireler = cv2.HoughCircles(gri_resim, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=0, maxRadius=0)
        
        return daireler
    
    def isaretle_ve_goster(self, resim, gozler):
        if gozler is not None:
            gozler = np.uint16(np.around(gozler))
            for daire in gozler[0, :]:
                merkez = (daire[0], daire[1])
                yari_cap = daire[2]
                cv2.circle(resim, merkez, 1, (0, 100, 100), 3)
                cv2.circle(resim, merkez, yari_cap, (255, 0, 255), 3)
            
            cv2.imshow('Göz Tespiti', resim)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Göz bulunamadı!")
            
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
