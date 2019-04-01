# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import time

import numpy as np
import tensorflow as tf
import cv2
import face_recognition
import collections
import csv


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_file(input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    frame = tf.placeholder("float32", name="frame")
    dims_expander = tf.expand_dims(frame, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    return normalized

def is_Int(s):
    try: 
        s = int(s)
        return True
    except ValueError:
        return False

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())

    return label

def largestFace(faces_coords):
    if (len(faces_coords) == 0):
        return None
    i = 0
    max = 0
    # Choose the biggest face
    for (top, right, bottom, left) in faces_coords:
        if bottom - top > max:
            face = faces_coords[i]
            max = bottom - top
        i += 1

    return face


if __name__ == "__main__":
    source = 0
    if is_Int(source):
        source = int(source)
    model = "mobilnet_v2_050_96_500000"
    dimss = 96
    if is_Int(dimss): 
        dimss = int(dimss)
    else:
        raise ValueError('Dimentions must be integers')
    label_file = 'tf_files/labels.txt'
    model_file = 'tf_files/' + model + '.pb'
    input_height = dimss
    input_width = dimss
    input_layer = "Placeholder"
    output_layer = "final_result"

    graph = load_graph(model_file)

    with tf.Session(graph=graph) as sess:
        collection = []
        neg = collections.deque(maxlen=5)
        neg.append(1.0)
        neu = collections.deque(maxlen=5)
        neu.append(1.0)
        pos = collections.deque(maxlen=5)
        pos.append(1.0)
        collection.append(neg)
        collection.append(neu)
        collection.append(pos)   
        face_locations = []
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        vs = cv2.VideoCapture(source)
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (100,100))
        labels = load_labels(label_file)
        input_operation = sess.graph.get_operation_by_name(input_name)
        output_operation = sess.graph.get_operation_by_name(output_name)
        img = read_tensor_from_file(input_height=input_height,
                                          input_width=input_width)
        try:
            with open("database.csv", "w+") as f1:
                writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
                row = ['neg', 'neu', 'pos']
                writer.writerow(row)
                frame_rate = 15
                prev = 0
                while True:
                    time_elapsed = time.time() - prev
                    ret, frame = vs.read()

                    if time_elapsed > 1./frame_rate:
                        prev = time.time()

                        frame = cv2.resize(frame, (400,300))
                        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB)

                        face_locations = face_recognition.face_locations(frame2)
                        face = largestFace(face_locations)
                        faceimg = np.zeros((100,100))
                        
                        if face:
                            frame2 = frame2[face[0]:face[2],face[3]:face[1]]
                            cv2.rectangle(frame, (face[3], face[0]), (face[1], face[2]), (0, 0, 255), 2)
                            frame2 = cv2.resize(frame2, (48,48))
                            faceimg = cv2.resize(frame2, (100,100))
                            out.write(faceimg)
                            t = sess.run(img, feed_dict={"frame:0": frame2})
                            results = sess.run(output_operation.outputs[0],
                                            {input_operation.outputs[0]: t})
                            results = np.squeeze(results)
                            top_k = results.argsort()[-5:][::-1]

                            collection[1].append(results[1])
                            collection[0].append(results[0])
                            collection[2].append(results[2])
                            row = [sum(collection[0])/5, sum(collection[1])/5, sum(collection[2])/5]
                            writer.writerow(row)
                            if sum(collection[0])/5 > 0.3:
                                cv2.putText(frame, 'NEGATIVE', (len(frame), 20 + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
                            elif sum(collection[1])/5 > 0.5:
                                cv2.putText(frame, 'NEUTRAL', (len(frame), 20 + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2)
                            else:
                                cv2.putText(frame, 'POSITIVE', (len(frame), 20 + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
                            for i in range(0,3):
                                cv2.putText(frame, labels[i], (10, i * 20 + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                                cv2.rectangle(frame, (130, i * 20 + 10),
                                                    (130 +int(sum(collection[i])/5 * 100),
                                                    (i + 1) * 20 + 4), (255, 0, 0), -1)
                                print(labels[i], results[i])
                        cv2.imshow('Video', frame)
                        cv2.imshow('Face', faceimg)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            vs.release()
                            out.release()
                            break
        except KeyboardInterrupt:
            pass
