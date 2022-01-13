import io

from flask import Flask, request, make_response, render_template, url_for, request, abort

from app import Utils
from app import service
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import csv
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename

api = Flask(__name__)

path_slice = "C:/Users/user/PycharmProjects/endend/app/static/dataset/sliced"
path_source = "C:/Users/user/PycharmProjects/endend/app/static/dataset/source"

api.config['MYSQL_HOST'] = 'localhost'
api.config['MYSQL_USER'] = 'root'
api.config['MYSQL_PASSWORD'] = ''
api.config['MYSQL_DB'] = 'db_owner'
api.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(api)

PHOTO_FOLDER = os.path.join('static','upload')

UPLOAD_FOLDER = 'app/static/upload/'
api.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api.config['PHOTO_FOLDER'] = PHOTO_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@api.errorhandler(500)
def internal_server_error(e):
    # note that we set the 500 status explicitly
    return render_template('500.html'), 500

@api.route('/messages/<int:idx>')
def message(idx):
    messages = ['Message Zero', 'Message One', 'Message Two']
    try:
        return render_template('message.html', message=messages[idx])
    except IndexError:
        abort(404)

@api.route('/')
def index():
    return render_template('home.html')

@api.route('/capture', methods=['GET'])
def capture():
    return render_template('index.html')

@api.route('/capture_img', methods=['POST'])
def capture_img():
    msg = service.save_img(request.form["img"])
    return make_response(msg)

@api.route('/identify', methods=['GET'])
def identify():
    return render_template('identify.html')

@api.route('/my-link/')
def my_link():
    img = cv2.imread('C:/Users/user/PycharmProjects/endend/app/static/dataset/source/0002.jpg')
    yolo = cv2.dnn.readNet("C:/Users/user/PycharmProjects/endend/app/yolo-license-plate-detection/model.weights",
                           "C:/Users/user/PycharmProjects/endend/app/yolo-license-plate-detection/darknet-yolov3.cfg")
    classes = []

    with open("C:/Users/user/PycharmProjects/endend/app/yolo-license-plate-detection/classes.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

    colorRed = (0, 0, 255)
    colorGreen = (0, 255, 0)
    colorWhite = (255, 255, 255)

    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    x, y, w, h = boxes[0]
    label = str(classes[class_ids[0]])
    cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
    cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 3, colorWhite, 2)
    height, width = img.shape[:2]
    resized_image = cv2.resize(img, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    location = None
    location = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
    print(location)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    plt.axis("off")
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite("C:/Users/user/PycharmProjects/endend/app/static/dataset/sliced/cropped.jpg", cropped_image)

    path_slice = "C:/Users/user/PycharmProjects/endend/app/static/dataset/sliced"
    for file_name in sorted(os.listdir(path_slice)):
        image = cv2.imread(os.path.join(path_slice, file_name))
        down_width = 300
        down_height = 100
        down_points = (down_width, down_height)
        resized_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow(resized_down)
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
                x = class_names[np.argmax(score)]

        def listToString(s):
            # initialize an empty string
            str1 = ""

            # traverse in the string
            for ele in s:
                str1 += ele

                # return string
            return str1

        myplate = listToString(num_plate)
        len_myplate = len(myplate)

        if len_myplate > 1 and len_myplate < 9 :
            return render_template('readPlat.html', plate=myplate)
        else:
            return render_template('ulangPlat.html')

@api.route('/owner', methods=['POST'])
def owner():
    data = []
    with open("C:/Users/user/PycharmProjects/endend/app/static/result/db.csv") as csvfile:
        reader = csv.reader(csvfile)
        hasil = request.form['txt']
        print(hasil)
        for row in reader:
            data.append(row)

    col = [x[0] for x in data]

    if hasil in col:
        for x in range(0, len(data)):
            if data[x][0] == hasil:
                owner = data[x]
                owner0 = f'PLAT     : {owner[0]}'
                owner1 = f'NAMA     : {owner[1]}'
                owner2 = f'ALAMAT   : {owner[2]}'
                owner3 = f'JABATAN  : {owner[3]}'
                owner4 = f'TELEPON  : {owner[4]}'
                print(owner)
                print(f'PLAT: {owner[0]}\nNAMA: {owner[1]}\nALAMAT: {owner[2]}\nJABATAN: {owner[3]}\nTELEPON: {owner[4]}')
                return render_template('owner.html', owner0=owner0, owner1=owner1, owner2=owner2, owner3=owner3,owner4=owner4)
    else:
        print("plat tidak ditemukan")
    return render_template('not_owner.html')

@api.route('/form')
def form():
    return render_template('testdb.html')

@api.route('/db', methods=['POST', 'GET'])
def db():
    if request.method == 'GET':
        return "Login via the login Form"

    if request.method == 'POST':
        id = 0
        plat = request.form['plat']
        nama = request.form['nama']
        ttl = request.form['ttl']
        alamat = request.form['alamat']
        no_telp = request.form['no_telp']
        files = request.files.getlist('files[]')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(api.config['UPLOAD_FOLDER'],filename))
        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO pemilik VALUES(%s,%s,%s,%s,%s,%s,%s)''', (id, plat, nama, ttl, alamat, no_telp,filename))
        mysql.connection.commit()
        cursor.close()
        return f"Done!!"

@api.route('/cek', methods=['POST', 'GET'])
def cek():
    hasil = request.form['txt']
    cursor = mysql.connection.cursor()
    cursor1 = mysql.connection.cursor()
    query = ("SELECT * FROM `pemilik` WHERE plat=" + "'" + hasil + "'");
    query2 = ("SELECT foto FROM `pemilik` WHERE plat=" + "'" + hasil + "'");
    cursor.execute(query)
    cursor1.execute(query2)
    result = cursor.fetchall()
    result2 = cursor1.fetchall()
    cursor.close()
    import json
    s = json.dumps(result2)
    x = s.replace("foto","")
    x = x.replace("[", "")
    x = x.replace("{", "")
    x = x.replace(":", "")
    x = x.replace("]", "")
    x = x.replace("}", "")
    x = x.replace(" ", "")
    x = x.replace('""', '')
    fix = x.replace('"', '')
    file = open("C:/Users/user/PycharmProjects/endend/app/static/result/profilpemilik.txt", "w")
    file.write(fix)
    file.close()
    fixfix = os.path.join(api.config['PHOTO_FOLDER'], fix)
    print(fix)
    print(result)
    print(result2)
    return render_template('hasilcoba.html', value=result, fixfix=fixfix)



