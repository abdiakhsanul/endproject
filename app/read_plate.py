import cv2
import Utils
import sys
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Folder untuk menyimpan dataset
path_slice = "static/dataset/sliced"
path_source = "static/images"

lic_data = cv2.CascadeClassifier('C:/Users/user/PycharmProjects/endend/app/static/dataset/haarcascade.xml')


path_slice = "C:/Users/user/PycharmProjects/endend/app/static/dataset/sliced"
for file_name in sorted(os.listdir(path_slice)):
  image = cv2.imread(os.path.join(path_slice, file_name))
  down_width = 300
  down_height = 100
  down_points = (down_width, down_height)
  resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
  #cv2.imshow(resized_down)
  cv2.imwrite("C:/Users/user/PycharmProjects/endend/app/static/dataset/sliced/cropped.jpg", resized_down)
  path_plate = "C:/Users/user/PycharmProjects/endend/app/static/dataset/sliced"

  # Looping file di direktori
  for name_file in sorted(os.listdir(path_plate)):
      src = cv2.imread(os.path.join(path_plate, name_file))
      blurred = src.copy()
      gray = blurred.copy()

      # Filtering
      for i in range(10):
          blurred = cv2.GaussianBlur(src, (5, 5), 0.5)

  # Ubah ke grayscale
  gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

  # Image binary
  ret, bw = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  print(ret, bw.shape)
  cv2.imwrite("segmentasi-bw.jpg", bw)
  # cv2_imshow(erode)
  # cv2.waitKey()

  erode = cv2.erode(bw.copy(), cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 6)))
  cv2.imwrite("segmentasi-erode.jpg", erode)
  # cv2.imshow("erode", erode)
  # cv2.waitKey()

  # Ekstraksi kontur
  contours, hierarchy = cv2.findContours(erode.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  # Looping contours untuk mendapatkan kontur yang sesuai
  for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      ras = format(w / h, '.2f')
      # print("x={}, y={}, w={}, h={}, rasio={}".format(x, y, w, h, ras))
      if h >= 40 and w >= 10 and float(ras) <= 1:
          # Gambar segiempat hasil segmentasi warna merah
          cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
          print("+ x={}, y={}, w={}, h={}, rasio={}".format(x, y, w, h, ras))
  cv2.imwrite("segmentasi-result.jpg", src)
  Cropped_img = cv2.imread('segmentasi-result.jpg')
  cv2.waitKey()

  ret, bw = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  contours_plate, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # index contour yang berisi kandidat karakter
  index_chars_candidate = []  # index

  # index counter dari setiap contour di contours_plate
  index_counter_contour_plate = 0  # idx

  # duplikat dan ubah citra plat dari gray dan bw ke rgb untuk menampilkan kotak karakter
  img_plate_rgb = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
  img_plate_bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

  for contour_plate in contours_plate:

      # dapatkan lokasi x, y, nilai width, height dari setiap kontur plat
      x_char, y_char, w_char, h_char = cv2.boundingRect(contour_plate)

      # Dapatkan kandidat karakter jika:
      #   tinggi kontur dalam rentang 40 - 60 piksel
      #   dan lebarnya lebih dari atau sama dengan 10 piksel
      if h_char >= 40 and h_char <= 60 and w_char >= 10:
          # dapatkan index kandidat karakternya
          index_chars_candidate.append(index_counter_contour_plate)

          # gambar kotak untuk menandai kandidat karakter
          cv2.rectangle(img_plate_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)
          cv2.rectangle(img_plate_bw_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)

      index_counter_contour_plate += 1

  # tampilkan kandidat karakter
  cv2.imwrite("segmentasi-result.jpg", src)

  if index_chars_candidate == []:

      # tampilkan peringatan apabila tidak ada kandidat karakter
      print('Karakter tidak tersegmentasi')
  else:
      score_chars_candidate = np.zeros(len(index_chars_candidate))

      # untuk counter index karakter
      counter_index_chars_candidate = 0

      # bandingkan lokasi y setiap kandidat satu dengan kandidat lainnya
      for chars_candidateA in index_chars_candidate:

          # dapatkan nilai y dari kandidat A
          xA, yA, wA, hA = cv2.boundingRect(contours_plate[chars_candidateA])
          for chars_candidateB in index_chars_candidate:

              # jika kandidat yang dibandikan sama maka lewati
              if chars_candidateA == chars_candidateB:
                  continue
              else:
                  # dapatkan nilai y dari kandidat B
                  xB, yB, wB, hB = cv2.boundingRect(contours_plate[chars_candidateB])

                  # cari selisih nilai y kandidat A dan kandidat B
                  y_difference = abs(yA - yB)

                  # jika perbedaannya kurang dari 11 piksel
                  if y_difference < 11:
                      # tambahkan nilai score pada kandidat tersebut
                      score_chars_candidate[counter_index_chars_candidate] = score_chars_candidate[
                                                                                 counter_index_chars_candidate] + 1

                      # lanjut ke kandidat lain
          counter_index_chars_candidate += 1

      print(score_chars_candidate)

      # untuk menyimpan karakter
      index_chars = []

      # counter karakter
      chars_counter = 0

      # dapatkan karakter, yaitu yang memiliki score tertinggi
      for score in score_chars_candidate:
          if score == max(score_chars_candidate):
              # simpan yang benar-benar karakter
              index_chars.append(index_chars_candidate[chars_counter])
          chars_counter += 1

          img_plate_rgb2 = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

          # tampilkan urutan karakter yang belum terurut
          for char in index_chars:
              x, y, w, h = cv2.boundingRect(contours_plate[char])
              cv2.rectangle(img_plate_rgb2, (x, y), (x + w, y + h), (0, 255, 0), 5)
              cv2.putText(img_plate_rgb2, str(index_chars.index(char)), (x, y + h + 50), cv2.FONT_ITALIC, 2.0,
                          (0, 0, 255), 3)

          # tampilkan karakter yang belum terurut
          # cv.imshow('Karakter Belum Terurut', img_plate_rgb2)

          # Mulai mengurutkan

          # untuk menyimpan koordinat x setiap karakter
          x_coors = []

          for char in index_chars:
              # dapatkan nilai x
              x, y, w, h = cv2.boundingRect(contours_plate[char])

              # dapatkan nilai sumbu x
              x_coors.append(x)

          # urutkan sumbu x dari terkecil ke terbesar
          x_coors = sorted(x_coors)

          # untuk menyimpan karakter
          index_chars_sorted = []

          # urutkan karakternya berdasarkan koordinat x yang sudah diurutkan
          for x_coor in x_coors:
              for char in index_chars:

                  # dapatkanx nilai koordinat x karakter
                  x, y, w, h = cv2.boundingRect(contours_plate[char])

                  # jika koordinat x terurut sama dengan koordinat x pada karakter
                  if x_coors[x_coors.index(x_coor)] == x:
                      # masukkan karakternya ke var baru agar mengurut dari kiri ke kanan
                      index_chars_sorted.append(char)
                      img_plate_rgb3 = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

                      # Gambar kotak untuk menandai karakter yang terurut dan tambahkan teks urutannya
                      for char_sorted in index_chars_sorted:
                          # dapatkan nilai x, y, w, h dari karakter terurut
                          x, y, w, h = cv2.boundingRect(contours_plate[char_sorted])

                          # gambar kotak yang menandai karakter terurut
                          cv2.rectangle(img_plate_rgb3, (x, y), (x + w, y + h), (0, 255, 0), 5)

                          # tambahkan teks urutan karakternya
                          cv2.putText(img_plate_rgb3, str(index_chars_sorted.index(char_sorted)), (x, y + h + 50),
                                      cv2.FONT_ITALIC, 2.0, (0, 0, 255), 3)

      img_height = 40
      img_width = 40
      # klas karakter
      class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                     'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

      # load model yang sudah terlatih
      model = keras.models.load_model('C:/Users/user/PycharmProjects/endend/app/static/my_model')

      # untuk menyimpan string karakter
      num_plate = []

      for char_sorted in index_chars_sorted:
          x, y, w, h = cv2.boundingRect(contours_plate[char_sorted])

          # potong citra karakter
          char_crop = cv2.cvtColor(bw[y:y + h, x:x + w], cv2.COLOR_GRAY2BGR)

          # resize citra karakternya
          char_crop = cv2.resize(char_crop, (img_width, img_height))

          # preprocessing citra ke numpy array
          img_array = keras.preprocessing.image.img_to_array(char_crop)

          # agar shape menjadi [1, h, w, channels]
          img_array = tf.expand_dims(img_array, 0)

          # buat prediksi
          predictions = model.predict(img_array)
          score = tf.nn.softmax(predictions[0])

          num_plate.append(class_names[np.argmax(score)])
          print(class_names[np.argmax(score)], end='')

          # Gabungkan string pada list
      plate_number = ''
      for a in num_plate:
          plate_number += a

      # Hasil deteksi dan pembacaan
      cv2.putText(gray, plate_number, (x, y + 50), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)



