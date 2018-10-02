from flask import Flask, render_template, jsonify, url_for, redirect, request, flash
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from evaluation.predict_batch import *
from lipnet.lipreading.helpers import text_to_labels, labels_to_text
from lipnet.lipreading.aligns import Align
from keras import backend as K

def serve_model(weight_path = 'evaluation/models/unseen-weights178.h5'):
    global lipnet
    global decoder
    lipnet = LipNet(img_c=3, img_w=100, img_h=50, frames_n=75,
                absolute_max_string_len=32, output_size=28)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                  postprocessors=[labels_to_text, spell.sentence])

def get_align(align_url, video_length=75):
    align = Align(video_length, text_to_labels).from_file(align_url)
    return labels_to_text(align.padded_label.astype('int'))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..', 'common','predictors','shape_predictor_68_face_landmarks.dat')



def get_video(video_path):
    
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    #video.from_video(video_path)
    video.from_frames(video_path) 
   
    X_data       = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])
    
    return X_data, input_length

app = Flask(__name__, static_url_path = "/static")

@app.route("/show/<ind_id>/<play_id>", methods=["GET", "POST"])
def main(ind_id, play_id):
    global lipnet
    global decoder
    
    print (os.listdir()) 
    play_list = os.listdir('app/static/datasets/s' + str(ind_id))
    play_list = [i.split('.')[0] for i in play_list]

    play_nm = play_list[int(play_id)] 
    image_url = './datasets_img/s' + str(ind_id) + '/' + play_nm
    video_url = './datasets/s' + str(ind_id) + '/' + play_nm + '.mpg'
    align_url = './app/static/datasets/align/' + play_nm + '.align' 
    
    X_data, input_length = get_video('./app/static/' + image_url)
    
    K.clear_session()
    serve_model()
    y_pred         = lipnet.predict(X_data)
    result         = decoder.decode(y_pred, input_length)[0]
    
    align = get_align(align_url, input_length) 
    del lipnet
    del decoder
  
    return render_template('index.html', video_url=video_url, align=align, predicted=result)
