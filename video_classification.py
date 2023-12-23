import argparse
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_array(image_array,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    float_caster = tf.cast(image_array, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(
        dims_expander, [input_height, input_width]
    )
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    return sess.run(normalized)

def load_labels(label_file):
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    return [l.rstrip() for l in proto_as_ascii_lines]


def process_frame(frame, graph, input_operation, output_operation, labels):
    t = read_tensor_from_image_array(
        frame,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std
    )

    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })

    results = np.squeeze(results)
    top_k = results.argsort()[-1:][::-1]  # Select the top prediction

    label = labels[top_k[0]]
    confidence = results[top_k[0]]

    return label, confidence


def display_results(frame, label, confidence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if confidence > 0.8:
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red

    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), font, 1, color, 2, cv2.LINE_AA)
    cv2.imshow("Video Classification", frame)

if __name__ == "__main__":
    file_name = "C:/Users/KARTIK/Desktop/Tensorflow/big_buck_bunny_720p_10mb.mp4"
    model_file = "C:/Users/KARTIK/Desktop/Tensorflow/inception_v3_2016_08_28_frozen.pb"
    label_file = "C:/Users/KARTIK/Desktop/Tensorflow/imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"

    graph = load_graph(model_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    labels = load_labels(label_file)

    cap = cv2.VideoCapture(file_name)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = process_frame(frame, graph, input_operation, output_operation, labels)
        display_results(frame, label, confidence)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
