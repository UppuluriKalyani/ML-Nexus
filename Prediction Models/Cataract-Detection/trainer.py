import tensorflow as tf
import os
from utils import utils
from build_model import model_tools
import model_architecture
from tensorflow.python.client import device_lib
from config import *

print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

session=tf.Session()
#create Placeholders for images and labels
images_ph=tf.placeholder(tf.float32,shape=[None,height,width,color_channels])
labels_ph=tf.placeholder(tf.float32,shape=[None,number_of_classes])

#training happens here
def trainer(network,number_of_images):
    #find error like squared error but better
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=network,labels=labels_ph)

    #now minize the above error
    #calculate the total mean of all the errors from all the nodes
    cost=tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cost", cost)#for tensorboard visualisation

    #Now backpropagate to minimise the cost in the network.
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    #print(optimizer)
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_save_name, graph=tf.get_default_graph())
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=4)
    counter=0
    for epoch in range(epochs):
        tools = utils()
        for batch in range(int(number_of_images / batch_size)):
            counter+=1
            images, labels = tools.batch_dispatch()
            if images == None:
                break
            loss,summary = session.run([cost,merged], feed_dict={images_ph: images, labels_ph: labels})
            print('loss', loss)
            session.run(optimizer, feed_dict={images_ph: images, labels_ph: labels})

            print('Epoch number ', epoch, 'batch', batch, 'complete')
            writer.add_summary(summary,counter)
        saver.save(session, os.path.join(model_save_name))

if __name__=="__main__":
    tools=utils()
    model=model_tools()
    network=model_architecture.generate_model(images_ph,number_of_classes)
    print (network)
    number_of_images = sum([len(files) for r, d, files in os.walk("data")])
    trainer(network,number_of_images)



