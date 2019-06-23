from PIL import Image
import http.client, urllib.request, urllib.parse, urllib.error, base64, json
import cv2 as cv
import numpy as np


def display(winname, img):
    cv.moveWindow(winname, 500, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)


def print_text(json_data):
    img_color  = cv.imread('C:/dev/Object-detection/testfiles_ara/output/thr4.jpg')
    height, width = img_color.shape[:2]
    print(width)
    result = json.loads(json_data)
    print(len(result['regions']))
    for i in result['regions']:
        print(i)
        for w in i['lines']:
            for r in w['words']:
                print(r['boundingBox'])
                y, x, h, w = r['boundingBox'].split(',')
                x, y, w, h = list(map(int, [x, y, w, h]))
                x = width - x
                print(x)
                cv.rectangle(img_color, (x, y), (x - w, y + h), (0, 0, 255), 1)
                # box = np.array( [x, y + h] , [x, y], [x+w, y], [x+w, y+h] )
                # cv.drawContours(img_color, [box], -1, 7)  # blue



    display('img_color', img_color)

    return

# Project oxford api 호출 '
def ocr_project_oxford(header, params, data):
    conn = http.client.HTTPConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/vision/v2.0/ocr?%s" % params, data, header)
    response = conn.getresponse()
    data = response.read().decode()
    print(data+"/n")
    print_text(data)
    conn.close()
    return

if __name__ ==  '__main__' :
    headers={
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key' : 'f18179b320bb4f729fd5637062a3df4e'
    }
    params = urllib.parse.urlencode({
        #파라미터 요청
        'language':'ko',
        'detectOrientation':'true'
    })


    data=open('C:/dev/Object-detection/testfiles_ara/output/thr4.jpg', 'rb').read()


    try:
        image_file = 'C:/dev/Object-detection/testfiles_ara/output/thr4.jpg'
        im = Image.open(image_file)
        #im.show()
        ocr_project_oxford(headers, params, data)
    except Exception as e:
        print(e)

