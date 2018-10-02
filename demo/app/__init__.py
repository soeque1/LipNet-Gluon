from flask import Flask, render_template, jsonify, url_for, redirect, request, flash

app = Flask(__name__, static_url_path = "/static")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

weight_path = 'models/unseen-weights178.h5'

@app.route("/show", methods=["GET"])
def main():
    test=os.listdir()
    video_url = './datasets/s1/bbaf2n.mpg'
    align = 'b'
    predicted = 'c'	
    return render_template('index.html', video_url=video_url, align_url=align, predicted=predicted, test=test)
