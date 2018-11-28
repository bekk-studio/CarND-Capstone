from styx_msgs.msg import TrafficLight
from keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
	# load cascade classifier
	cascade_name = './light_classification/cNewBag16x32LBPw30d2_3.xml'
	self.cascade = cv2.CascadeClassifier(cascade_name)	
	
	tf.reset_default_graph()
        config = tf.ConfigProto(
           gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4),
           device_count = {'GPU': 1}
        )
        
        self.session = tf.Session(config=config);

        save_file = './light_classification/mixNetI-1.ckpt.meta'
        saver = tf.train.import_meta_graph(save_file)
        saver.restore(self.session,tf.train.latest_checkpoint('./light_classification'))

        graph = tf.get_default_graph()
        self.batch_x =  graph.get_tensor_by_name("batch_x:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.out = graph.get_tensor_by_name("out:0")

        self.ans = tf.argmax(self.out, 1);
	

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
	# traffic light detection	
	img = cv2.resize(image, (640, 480))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	TL_boxes = self.cascade.detectMultiScale(gray, 1.2, 1, 0, (16, 32), (100, 200))

        colors_hist = np.zeros(TrafficLight.UNKNOWN+1, dtype=int)

        for (x,y,w,h) in TL_boxes:
            p0 = (x, y)
            p1 = (x+w, y+h)
            tl_image = cv2.resize(img[p0[1]:p1[1], p0[0]:p1[0], :], (16, 32))
            
	    # Color classification
	    tl_image = (tl_image.astype('float32')-127.)/255.
            a = self.session.run([self.ans], feed_dict={self.batch_x: [tl_image],
                                                    self.keep_prob: 1.0})

            tl_color = a[0]
         
            colors_hist[tl_color] = colors_hist[tl_color] + 1
	
	colors_hist[TrafficLight.UNKNOWN] = 1
	colors_hist[TrafficLight.RED] = colors_hist[3]
	colors_hist[3] = 0
        result = np.argmax(colors_hist)

        return result
