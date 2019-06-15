# https://github.com/andreasntr/ColorClassifier/blob/master/getData.py

############
# Get Data #
############

from firebase import firebase
import json
db = firebase.FirebaseApplication("https://color-classification.firebaseio.com", None)
jsonData = db.get("/colors", None)
data = []

for user in jsonData.keys():
    entry = {}
    entry['r'] = jsonData[user]['r']
    entry['g'] = jsonData[user]['g']
    entry['b'] = jsonData[user]['b']
    entry['label'] = jsonData[user]['label']
    data.append(entry)
json.dump({"data":data}, open("data.json", "w"))


################
# process Data #
################

from firebase import firebase
import json
import numpy as np

data = json.load(open("data.json"))['data']
cols = []
lbls = []
labelsValues = [
    "red-ish",
    "green-ish",
    "blue-ish",
    "orange-ish",
    "yellow-ish",
    "pink-ish",
    "purple-ish",
    "brown-ish",
    "grey-ish"
]
for submission in data:
    color = []
    color.append(submission["r"] / 255)
    color.append(submission["g"] / 255)
    color.append(submission["b"] / 255)
    cols.append(color)
    lbls.append(labelsValues.index(submission["label"]))

colors = np.array(cols, dtype=np.float32)
labels = np.array(lbls, dtype=np.int8)
np.savez_compressed("processedData", colors = colors, labels = labels)


##############
## training ##
##############

import tensorflow as tf
import numpy as np
from random import randint
import json

colors = None
labels = None
data_size = 0

tf.enable_eager_execution()

with np.load("processedData.npz") as savedData:
    colors = np.array(savedData['colors'], dtype=np.float32)
    labels = tf.one_hot(savedData['labels'],9, dtype = tf.float32).numpy()
    data_size = len(savedData['colors'])

train_size = int(data_size*0.8)
test_size = validation_size = int((data_size - train_size)/2)

indexes = [randint(0, data_size-1) for i in range(train_size)]
colors_train = tf.constant([colors[i] for i in indexes])
labels_train = tf.constant([labels[i] for i in indexes])
test_indexes = []
for i in range(0, data_size):
    if not (i in indexes):
        test_indexes.append(i)
test_indexes = [test_indexes[randint(0, test_size-1)] for i in range(test_size)]
colors_test = tf.constant([colors[i] for i in test_indexes])
labels_test = tf.constant([labels[i] for i in test_indexes])
validation_indexes = []
for i in range(0, data_size):
    if not (i in test_indexes) and not (i in indexes):
        validation_indexes.append(i)
validation_indexes = [validation_indexes[randint(0, validation_size-1)] for i in range(validation_size)]
colors_validation = tf.constant([colors[i] for i in validation_indexes])
labels_validation = tf.constant([labels[i] for i in validation_indexes])

np.savez_compressed("dataset", train_x = colors_train.numpy(), train_y = labels_train.numpy(), test_x = colors_test.numpy(),
        test_y = labels_test.numpy(), validation_x = colors_validation.numpy(), validation_y = labels_validation.numpy())

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(3,),activation=tf.nn.relu),
    tf.keras.layers.Dense(9, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training:")
model.fit(colors_train, labels_train, epochs=10, batch_size=32)
print("Training ended. Validating:")
model.fit(colors_validation, labels_validation, epochs=10, batch_size=32)
json.dump({'model':model.to_json()}, open("model.json", "w"))
model.save_weights("model_weights.h5")


################
## test Model ##
################

import tensorflow as tf
import numpy as np
import json

tf.enable_eager_execution()

with np.load("dataset.npz") as savedData:
    colors_test = tf.constant(savedData['test_x'])
    labels_test = tf.constant(savedData['test_y'])

model = tf.keras.models.model_from_json(json.load(open("model.json"))["model"], custom_objects={})
model.load_weights("model_weights.h5")
predictions = model.predict(colors_test, batch_size=32, verbose=1)
predictions = tf.one_hot(np.argmax(predictions,1),9)
equals = np.sum(np.all(predictions.numpy()==labels_test.numpy(),axis=1))
print("Guess accuracy: {}".format(equals/len(colors_test.numpy())))


#################
# Predict color #
#################

import tensorflow as tf
import numpy as np
import json
from tkinter import *
import tkinter.messagebox as messagebox
from subprocess import run
from sys import exit

labelsValues = [
    "red-ish",
    "green-ish",
    "blue-ish",
    "orange-ish",
    "yellow-ish",
    "pink-ish",
    "purple-ish",
    "brown-ish",
    "grey-ish"
]

def updateModel():
    ans = messagebox.askquestion("Update model","The window will be closed and restarted when the process completes.\
    \nThe process will take a while...\nDo you wish to continue?")
    if ans == "yes":
        window.destroy()
        run("python getData.py")
        run("python processData.py")
        run("python train.py")
        run("python predict.py")
        exit(0)
    else:
        if(noModel):
            exit(0)

def predict():
    r = R.get()/255
    g = G.get()/255
    b = B.get()/255
    prediction_lbl.configure(text=labelsValues[np.argmax(model.predict(
        tf.constant([[r,g,b]], dtype=tf.float32)))])

def update(event):
    preview.configure(bg='#{:2x}{:2x}{:2x}'.format(R.get(),G.get(),B.get()).replace(" ", "0"))

tf.enable_eager_execution()
noModel = False

#LAYOUT
window = Tk()
windowWidth = window.winfo_reqwidth()
windowHeight = window.winfo_reqheight()
positionRight = int(window.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(window.winfo_screenheight()/2 - windowHeight/2)
window.geometry("+{}+{}".format(positionRight, positionDown))
window.geometry('200x170')
window.title("Color Classifier")

menu = Menu(window)
window.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label = "File", menu=fileMenu)
fileMenu.add_command(label = "Update model", command=updateModel)
preview = Label(window, width=10, height=5)
R_lbl = Label(window, text="R", fg='red')
G_lbl = Label(window, text="G", fg='green')
B_lbl = Label(window, text="B", fg='blue')
R = Scale(window, from_=0, to=255, orient=HORIZONTAL, fg='red', command=update)
G = Scale(window, from_=0, to=255, orient=HORIZONTAL, fg='green', command=update)
B = Scale(window, from_=0, to=255, orient=HORIZONTAL, fg='blue', command=update)
predictButton = Button(window,text="Predict", command = predict)
prediction_lbl = Label(window, text = "")
R_lbl.grid(row= 0, sticky=S)
G_lbl.grid(row= 1, sticky=S)
B_lbl.grid(row= 2, sticky=S)
R.grid(row= 0,column =1)
G.grid(row= 1,column =1)
B.grid(row= 2,column =1)
preview.configure(bg='#{:2x}{:2x}{:2x}'.format(R.get(),G.get(),B.get()).replace(" ", "0"))
predictButton.grid(columnspan=3)
prediction_lbl.grid(row = 0, column = 2, columnspan = 3)
preview.grid(row = 1, column = 2, sticky = N, rowspan = 3)
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)
window.columnconfigure(2, weight=1)

try:
    with open("model.json"):
        pass
except FileNotFoundError:
    noModel = True
    ans = messagebox.askquestion("No model found","Do you wish to start the creation of a new model?")
    if ans == "yes":
        updateModel()
    else:
        exit(0)

#LOADING MODEL
model = tf.keras.models.model_from_json(json.load(open("model.json"))["model"], custom_objects={})
model.load_weights("model_weights.h5")


window.mainloop()


