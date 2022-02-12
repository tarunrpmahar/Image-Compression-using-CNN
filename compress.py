from __future__ import division

from Tkinter import *
import os
from tkFileDialog import askdirectory
import Tkinter as tk
from PIL import ImageTk, Image
import tkFileDialog as filedialog
import tkMessageBox
class CanvasButton:
    def __init__(self, canvas):
        self.canvas = canvas
        self.number = tk.IntVar()
        self.button = tk.Button(canvas, text="Select a file",bg="black",fg="white",
                                command=self.buttonclicked)
        self.id = canvas.create_window(580,430, width=120, height=25,
                                       window=self.button,anchor=CENTER)
        self.button1 = tk.Button(canvas, text="Submit",bg="black",fg="white",
                                command=self.buttonclicked2)
        self.id = canvas.create_window(660, 470, width=100, height=25,
                                       window=self.button1, anchor=CENTER)
        self.button2 = tk.Button(canvas, text="Click to exit", bg="black", fg="white",
                                command=self.buttonclicked3)
        self.id = canvas.create_window(880, 500, width=90, height=25,
                                       window=self.button2, anchor=CENTER)
        self.button3 = tk.Button(canvas, text="Select Output Directory", bg="black", fg="white",
                                 command=self.buttonclicked4)
        self.id = canvas.create_window(760, 430, width=170, height=25,
                                       window=self.button3, anchor=CENTER)

    def buttonclicked(self):
        global my_image
        root.filename = filedialog.askopenfilename(initialdir="/home/tarun/Capstone", title="Select a file",
                                                   filetypes=(("png files", "*png"),("jpg files","*jpg"), ("all files", "*.*")))
        my_label = Label(root, text=root.filename).pack()
        my_image = ImageTk.PhotoImage(Image.open(root.filename))
        my_image_label = Label(image=my_image).pack()

    def buttonclicked4(self):
        Tk().withdraw()
        root.dirname = filedialog.askdirectory(initialdir=os.getcwd(), title='Please select a directory')
	print ("You chose %s" % root.dirname)

    def buttonclicked2(self):

        import os
        import sys
        import skimage.io
        import tensorflow as tf
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        from parameters import Or_Parameter
        from set_image import ld_one_image, normalization
        from model import ConvNet

        plt.style.use('ggplot')
        image = ld_one_image(root.filename)

        good_par = Or_Parameter(verbose=False)
        tf_img = tf.compat.v1.placeholder(tf.float32, [None, good_par.image_h, good_par.image_w, good_par.image_c],
                                          name="images")
        tf_cls = tf.compat.v1.placeholder(tf.int64, [None], name='class')

        cnn = ConvNet()
        if good_par.fine_tuning:
            cnn.ld_vgg_wts()

        last_cnn, space, P_cls = cnn.cnn_build(tf_img)
        binary_map = cnn.get_binary(tf_cls, last_cnn)

        with tf.compat.v1.Session() as sess:
            tf.compat.v1.train.Saver().restore(sess, good_par.model_path)
            last_cnn_value, P_cls_value = sess.run([last_cnn, P_cls], feed_dict={tf_img: image})

            all_binary_pred = P_cls_value.argsort(axis=1)

            msroi = None
            for i in range(-1 * good_par.top_k, 0):

                cur_cls = all_binary_pred[:, i]
                binary_val = sess.run(binary_map, feed_dict={tf_cls: cur_cls, last_cnn: last_cnn_value})
                norm_binary_map = normalization(binary_val[0])

                if msroi is None:
                    msroi = 1.2 * norm_binary_map
                else:
                    msroi = (msroi + norm_binary_map) / 2
            msroi = normalization(msroi)

        sess.close()

        figure, axis = plt.subplots(1, 1, figsize=(12, 9))
        axis.margins(0)
        plt.axis('off')
        plt.imshow(msroi, cmap=plt.cm.jet, interpolation='nearest')
        plt.imshow(image[0], alpha=0.4)

        if not os.path.exists('output'):
            os.makedirs('output')
        plt.savefig(root.dirname + '/output/Importance_map.png')
        skimage.io.imsave(root.dirname + '/output/msroi_binary_map.jpg', msroi)
        print("Both binary_map and Importance_map are created.")
#-------------------------------------------------------------------------------
        import numpy as np
        import argparse
        from PIL import Image
        import os

        arg_pars = argparse.ArgumentParser()

        arg_pars.add_argument('-jpeg_compression', type=int, default=50)
        arg_pars.add_argument('-model', type=int, default=1)
        arg_pars.add_argument('-single', type=int, default=1)
        arg_pars.add_argument('-print_metrics', type=int, default=0)
        arg_pars.add_argument('-image', type=str, default=root.filename)
        arg_pars.add_argument('-map', type=str, default='/home/adarsh/Capstone/output/msroi_binary_map.jpg')
        arg_pars.add_argument('-output_directory', type=str, default='/home/adarsh/Capstone/output')
        arg_pars.add_argument('-use_convert', type=int, default=0)

        args = arg_pars.parse_args()

        def qual_compr(native, new_sal):
            if args.print_metrics:
                print (args.image)

            if native.size != new_sal.size:
                new_sal = new_sal.resize(native.size)

            new_sal_arr = np.asarray(new_sal)
            qual_img = []
            qual_pace = [i * 10 for i in range(1, 11)]

            os.makedirs('chintan')
            for q in qual_pace:
                name = 'chintan/temp_' + str(q) + '.jpg'
                if args.use_convert:
                    os.system('lets see ' + str(q) + ' ' + args.image + ' ' + name)
                else:
                    native.save(name, quality=q)
                qual_img.append(np.asarray(Image.open(name)))
                os.remove(name)
            os.rmdir('chintan')

            k_hare = qual_img[-1][:]
            shape = k_hare.shape
            k_hare.flags.writeable = True
            q_inp = [np.percentile(new_sal_arr, j) for j in qual_pace]
            lower, medium, higher = 1, 5, 9

            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        new_ss = new_sal_arr[i, j]

                        if args.model == 1:
                            for indx, q_i in enumerate(q_inp):
                                if new_ss < q_i:
                                    new_qq = indx + 1
                                    break

                        else:
                            raise Exception("unknown model number")

                        if new_qq < lower: new_qq = lower
                        if new_qq > higher: new_qq = higher
                        k_hare[i, j, k] = qual_img[new_qq][i, j, k]

            cmprs = args.output_directory + '/' + 'Compressed_' + args.image.split('/')[-1] + '_' + str(
                args.jpeg_compression) + '.jpg'
            native.save(cmprs, quality=args.jpeg_compression)
            print('compressed image in your working directory')

        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

        if args.single:
            original = Image.open(args.image)
            sal = Image.open(args.map)
            qual_compr(original, sal)

        tkMessageBox.showinfo("Title", "compressed image is in desktop output directory")

    def buttonclicked3(self):
        root.destroy()
        exit()


root = tk.Tk()
root.title("Image Compression")
root.configure(background="light blue")
theLabel=Label(root,text="IMAGE COMPRESSION",bg="black",fg="white",font="none 30 bold")
theLabel.pack()
canvas=Canvas(width=1024,height=756)
imgpath = 'abb.jpg'
img = Image.open(imgpath)
img=img.resize((1290,662),Image.ANTIALIAS)
photo = ImageTk.PhotoImage(img)

canvas = tk.Canvas(root, bd=140, highlightthickness=10,width=3340,height=1440)
canvas.configure(background="light blue")
canvas.pack()
canvas.create_image(690, 342, image=photo)

CanvasButton(canvas)

root.mainloop()
