import os
import shutil
import time

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from skimage import img_as_ubyte
from tqdm import tqdm

import calculate_PSNR_20181008 as com_y_psnr_
import utils
from common.layers import conv2d_weight_norm


class wedsr(object):

    def __init__(   self,img_size=48,
                    num_layers=32,
                    feature_size=32,
                    scale=2,
                    output_channels=3):

        def _ignore_boundary(images,scale):
            boundary_size = scale + 6
            images = images[:, boundary_size:-boundary_size, boundary_size:-boundary_size, :]
            return images

        def _float32_to_uint8(images):
            images = images * 255.0
            images = tf.round(images)
            images = tf.saturate_cast(images, tf.uint8)
            return images

        def _residual_block(x, feature_size):
            skip = x
            x = conv2d_weight_norm(
                x,
                feature_size * 4,
                3,
                padding='same',
                name='conv0',
            )
            x = tf.nn.relu(x)
            x = conv2d_weight_norm(
                x,
                feature_size,
                3,
                padding='same',
                name='conv1',
            )
            return x + skip

        print("Begin creating wEDSR...")
        self.img_size = img_size                    #缩小的图像块，这里是48
        self.num_layers = num_layers                #层数
        self.scale = scale                          #x2/x3/x4
        self.output_channels = output_channels      #3->RGB
        self.feature_size = feature_size            #输出卷积维度大小
        self.feature_size_first=feature_size*4      #输入卷积维度大小

        #Placeholder for image inputs
        self.input = x = tf.placeholder(tf.uint8,[None,img_size,img_size,output_channels])
        #Placeholder for upscaled image ground-truth
        self.target = y = tf.placeholder(tf.uint8,[None,img_size*scale,img_size*scale,output_channels])

        x=tf.cast(x,tf.float32)
        y=tf.cast(y,tf.float32)

        mean_x = 127#tf.reduce_mean(self.input)
        image_input =x- mean_x
        mean_y = 127#tf.reduce_mean(self.target)
        image_target =y- mean_y

        skip=image_input

    #     One convolution before res blocks and to convert to required feature depth
        x = slim.conv2d(image_input,self.feature_size,[3,3],)


        #Add the residual blocks to the model
        for i in range(num_layers):
            x = utils.dirac_conv2d(x,self.feature_size,3,3,1,1,name="dircov_"+str(i)+"_1",atrainable=True)
            x = tf.nn.relu(x, name="relu_"+str(i))
            # x = utils.dirac_conv2d(x,feature_size,3,3,1,1,name="dircov_"+str(i)+"_2",atrainable=True)
            # x = tf.nn.relu(x, name="relu_1_"+str(i))
            # x = utils.dirac_conv2d_wtf(x,feature_size_first,3,3,1,1,name="dircov_"+str(i)+"_1")
            # x = tf.nn.relu(x, name="relu_"+str(i))    
            # x = utils.dirac_conv2d_wtf(x,feature_size,3,3,1,1,name="dircov_"+str(i)+"_2")
            
        
        #Upsample output of the convolution        
        x = utils.upsample(x,scale,feature_size,None)
        skip = slim.conv2d(skip,self.feature_size,[3,3],)
        skip = utils.upsample(skip,scale,self.feature_size,None,filtersize=[5,5])


        # x = slim.conv2d(image_input,self.feature_size_first,(3,3), activation_fn=tf.nn.relu)
        # skip = utils.upsample(x,scale,self.feature_size_first,activation=tf.nn.relu,filtersize=[5,5])

        # for i in range(self.num_layers):
        #     x = utils.dirac_conv2d(x,self.feature_size_first,3,3,1,1,name="dircov_"+str(i)+"_1",atrainable=True)
        #     x = tf.nn.relu(x, name="relu_"+str(i)) 
        #     x = utils.dirac_conv2d(x,feature_size,3,3,1,1,name="dircov_"+str(i)+"_2",atrainable=True)
        #     # x = tf.nn.relu(x, name="relu_"+str(i)+"_2")  
        #     # x = utils.dirac_conv2d_wtf(x,self.feature_size_first,3,3,1,1,name="dircov_"+str(i)+"_1")
        #     # x = tf.nn.relu(x, name="relu_"+str(i)+"_1")    
        #     # x = utils.dirac_conv2d_wtf(x,self.feature_size,3,3,1,1,name="dircov_"+str(i)+"_2")
        #     # x = tf.nn.relu(x, name="relu_"+str(i)+"_2")    
        # x = utils.upsample(x,scale,feature_size,activation=tf.nn.relu,filtersize=[3,3])

        '''
        with tf.variable_scope('skip'):
            # skip = slim.conv2d(image_input,self.feature_size,(3,3), activation_fn=tf.nn.relu)
            skip = utils._subpixel_block(   x,
                                            kernel_size = (5,5),
                                            feature_size= self.output_channels,
                                            scale=self.scale)

        # with tf.variable_scope('input'):
        #     x = conv2d_weight_norm( inputs=image_input,
        #                             filters=self.feature_size,
        #                             kernel_size=3,
        #                             padding='same',
        #                             trainable=True,
        #                             activation=tf.nn.relu)
        for i in range(self.num_layers):
            with tf.variable_scope('layer{}'.format(i)):
                x = _residual_block(x, feature_size=self.feature_size)

        with tf.variable_scope('output'):
            x = utils._subpixel_block(  x, 
                                        kernel_size =(3,3),
                                        feature_size=self.output_channels,
                                        scale=self.scale)
        '''

        x_target = skip + x
        x_target = tf.maximum(x_target,-127)
        x_target = tf.minimum(x_target,128)

        self.out = tf.saturate_cast(x_target+mean_x,tf.uint8)

        # self.loss = tf.reduce_mean(tf.losses.absolute_difference(image_target,x_target))
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(image_target,x_target))

        
        psnr = tf.image.psnr(_ignore_boundary(self.target,self.scale),_ignore_boundary(self.out,self.scale),max_val=255,)
        self.PSNR = tf.reduce_mean(psnr)
        ssim = tf.image.ssim(_ignore_boundary(self.target,self.scale),_ignore_boundary(self.out,self.scale),max_val=255,)
        self.SSIM = tf.reduce_mean(ssim)
        # self.SSIM = tf.metrics.mean(ssim)

        self.bicubics = tf.image.resize_images(self.input, (self.img_size*self.scale, self.img_size*self.scale),tf.image.ResizeMethod.BICUBIC)
        self.bicubics=tf.clip_by_value(self.bicubics,0.0,255.0)
        self.bicubics=tf.cast(self.bicubics,tf.uint8)
        bicupsnr = tf.image.psnr(_ignore_boundary(self.target,self.scale),_ignore_boundary(self.bicubics,self.scale),max_val=255,)
        self.BIC_PSNR = tf.reduce_mean(bicupsnr)

        tf.summary.scalar("loss",self.loss)
        tf.summary.scalar("Out_PSNR",self.PSNR)
        tf.summary.scalar("ReB_PSNR",self.BIC_PSNR)
        tf.summary.scalar("SSIM",self.SSIM)
    
        #Image summaries for input, target, and output
        tf.summary.image("input_image",tf.cast(self.input,tf.uint8))
        tf.summary.image("resize_image",tf.cast(self.bicubics,tf.uint8))
        tf.summary.image("target_image",tf.cast(self.target,tf.uint8))
        tf.summary.image("output_image",tf.cast(self.out,tf.uint8))

        #Tensorflow graph setup... session, saver, etc.
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver(max_to_keep=3)
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['Conv_5','Conv_6','Conv_4','Conv_3','Conv_7'])
        self.saver_change = tf.train.Saver(variables_to_restore)
        
        print("Model creation completed!")
    
    """
    Train the neural network
    """
    def train(self,iterations=1000,savedir="saved_models",scale=2,batchsize=8,layers=16,start_learning_rate=0.0001,trainset="train",testset="test",log_dir='logs',changesize=False):

        #shutil.rmtree(save_dir)
        #Just a tf thing, to merge all summaries into one
        
        all_step = tf.Variable(0, name='all_step', trainable=False)
        learning_rate = tf.train.exponential_decay(start_learning_rate,
                                                   global_step=all_step, 
                                                   decay_steps=20000,
                                                   decay_rate=0.99, 
                                                   staircase=False)#lr=slr*dr^(i/ds)
        add_global = all_step.assign_add(1)
        tf.summary.scalar("Learn_rate",learning_rate)
        merged = tf.summary.merge_all()
        #Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate)
#        optimizer=tf.train.GradientDescentOptimizer(learning_rate)
               
        #This is the train operation for our objective
        train_op = optimizer.minimize(self.loss)

        trainimgout = utils.read_records(trainset,self.img_size*scale, self.img_size*scale,self.img_size,shuffle=True) # 读取函数
        trainbatch=tf.train.batch(trainimgout,batchsize)#训练集输入
        
#        testimgout = utils.read_records(testset,self.img_size*scale, self.img_size*scale,self.img_size,shuffle=False) # 读取函数
#        testbatch=tf.train.batch(testimgout,1)#测试集输入
        
        # testimg16=utils.read_records(testset,self.img_size*scale, self.img_size*scale,self.img_size,shuffle=False)
        # testbatch16 = tf.train.batch(testimg16,1)
        # test_num=16
        # tests_batch=testbatch16
        
        testimg_set14=utils.read_records(testset,self.img_size*scale, self.img_size*scale,self.img_size,shuffle=False)
        testbatch_set14 = tf.train.batch(testimg_set14,1)
        tests_batch=testbatch_set14

        if self.scale == 2:
            test_num=279
        elif self.scale == 3:
            test_num=108
        elif self.scale == 4:
            test_num=51

        #test参数设置
        #每过多少步测试一次
        testepoch = test_num
        max_psnr = 28.0

        testnp_mem = np.empty((4,0), dtype = np.float16)
        eve_test   = np.zeros((4,1), dtype = np.float16)
        last_test_num = 0
        
        cond1=os.path.exists(savedir)#有文件夹
        cond2=os.path.exists(savedir+'/checkpoint')#有之前的checkpoint

        #Operation to initialize all variables
        init = tf.global_variables_initializer()

        print("Begin training...")
        with self.sess as sess:
            sess.run(init)
            #判断是否有文件夹
            if cond1 :
                if cond2:
                    if changesize:
                        self.saver_change.restore(self.sess,tf.train.latest_checkpoint(savedir))
                    else:
                        self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))

                    np_savedir=savedir+'/outfile.npy'
                    try:
                        testnp_mem = np.load(np_savedir)
                        last_test_num=testnp_mem[0][-1]
                    except:
                        pass
                    print('='*50)
                    print('restore success!')
                    print('='*50)
                elif not(cond2):
                    shutil.rmtree(savedir)
    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
                            
            #create summary writer for train
            train_writer = tf.summary.FileWriter(savedir+"/train",sess.graph)
            test_writer  = tf.summary.FileWriter(savedir+"/test" ,sess.graph)

#            disp_num=int(iterations/10)
#            lisl=[j*disp_num for j in range(1,11)]#每1/10存储一次
            variable_names = [v.name for v in tf.trainable_variables()]
            print(variable_names)
            # wight_mem = np.zeros((layers,4), dtype = np.float32)
            # for i in range(layers):
            #     a1_name='dircov_'+str(i)+'_1'+'/alpha:0'
            #     b1_name='dircov_'+str(i)+'_1'+'/beta:0'
            #     a2_name='dircov_'+str(i)+'_2'+'/alpha:0'
            #     b2_name='dircov_'+str(i)+'_2'+'/beta:0'
            #     wight_mem[i]=sess.run([a1_name,b1_name,a2_name,b2_name])
            # print(wight_mem)

            for i in range(iterations):

                trainsm,trainbi =sess.run(trainbatch)
                feed = {self.input:trainsm,self.target:trainbi}           
                _=sess.run(add_global)
                #Run the train op and calculate the train summary
                summary,_,loss,psnr,ssim = sess.run([merged,train_op,self.loss,self.PSNR,self.SSIM],feed)
                train_writer.add_summary(summary,i)

                if (i+1)%2  == 0:
                    print('epoch:'   ,'%07d' %(i+1), 
                          ' | loss:' ,'{:.5f}'.format(loss) ,
                          ' | psnr:' ,'{:.4f}'.format(psnr) , 
                          ' | ssim:','{:.4f}'.format(ssim) ,
                          ' | lear:' ,'{:.9f}'.format(sess.run(learning_rate)))

#                if (i+1)%testepoch == 0:
#                    for img in img_files:
#                    tmp = imageio.imread(image_dir+"/"+img)


                if (i+1)%testepoch == 0:
                    eve_test[0] = (i+1)/testepoch + last_test_num
                    testloss   = 0.0
                    testpsnr   = 0.0
                    testssim   = 0.0
                    resizeps   = 0.0
                    

                    for j in range(test_num):
                        testsm,testbi =sess.run(tests_batch)

                        test_feed = {self.input:testsm,self.target:testbi}
                        tsummary,tloss,tpsnr,tssim,rpsnr = sess.run([merged,self.loss,self.PSNR,self.SSIM,self.BIC_PSNR],test_feed)
                        test_writer.add_summary(tsummary,i+j-test_num)

                        testloss   += tloss
                        testpsnr   += tpsnr
                        testssim   += tssim
                        resizeps   += rpsnr

                    eve_test[1]  = testloss/test_num
                    eve_test[2]  = testpsnr/test_num
                    eve_test[3]  = testssim/test_num


                    print('testepoch:'  ,'%06d' %((i+1)/testepoch),
                          ' | tloss:'   ,'{:.5f}'.format(eve_test[1][0]),
                          ' | tpsnr:'   ,'{:.4f}'.format(eve_test[2][0]),
                          ' | tssim:'   ,'{:.4f}'.format(eve_test[3][0]),
                          ' | rpsnr:'   ,'{:.4f}'.format(resizeps/test_num))  
                    print('-'*50)

                    if eve_test[2][0] <60 and eve_test[2][0] > max_psnr:
                        max_psnr = eve_test[2][0]
                        save_dirs=savedir+'/model_'+str((i+1)/testepoch + last_test_num)+'psnr_'+'{:.4f}'.format(max_psnr)
                        self.saver.save(sess,save_dirs)
                        print('Save model!\tTime:',time.asctime(time.localtime(time.time())))
                        print('*'*50)
                    
                    testnp_mem = np.append(testnp_mem, eve_test,axis = 1)
                    np.save(savedir+'/outfile',testnp_mem)


            print('Model training has been completed!')
            save_dirs=savedir+'/model_last_psnr_'+'{:.4f}'.format(eve_test[2][0])
            self.saver.save(sess,save_dirs)
                    
            coord.request_stop()
            coord.join(threads)

    def predict_sets(self,chepdir,photodir,inputdir,outputdir,testpdir):
#network.predict(flags.checkpoint_url,
#                flags.photodir,
#                flags.inputdir,
#                flags.outputdir)
        test_mem = np.empty((7,0))#, dtype = float
        eve_test = np.zeros((7,1))
        num=0
        
        print("Predicting...")
        init = tf.global_variables_initializer()
        with self.sess as sess:
            sess.run(init)
            self.saver.restore(sess,tf.train.latest_checkpoint(chepdir))
            
            image_dir=photodir
            img_files = os.listdir(image_dir)
            for img in img_files:
                img_is_GRAY = False
                input_img = imageio.imread(image_dir+"/"+img)
                name=img.split('.')[0]+'x{}.'.format(self.scale)+img.split('.')[1]
                try:
                    input_tmp=imageio.imread(testpdir+"/"+name)
                except:
                    print('not found {}!'.format(name))
                if len(input_tmp.shape) == 2:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                    input_tmp = cv2.cvtColor(input_tmp, cv2.COLOR_GRAY2RGB)
                    img_is_GRAY =True

                if (input_tmp.shape[0] == self.img_size and input_tmp.shape[1] == self.img_size):
                    output_tmp = sess.run(self.out,feed_dict={self.input:[input_tmp]})[0]
                else :
                    sr_size=int(self.img_size)#32
                    sr_num=int(self.img_size/2)
                    num_down = int(input_tmp.shape[0]//sr_num)-1
                    num_across = int(input_tmp.shape[1]//sr_num)-1
                    #tmp_image = np.zeros([input_tmp.shape[0]*self.scale,input_tmp.shape[1]*self.scale,3])
                    tmp_image = np.zeros([input_img.shape[0],input_img.shape[1],3])
                    
                    for i in tqdm(range(0,num_down)):
                        for j in range(0,num_across):
                            tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[sr_size*i>>1:sr_size*(i+2)>>1,sr_size*j>>1:sr_size*(j+2)>>1]]})[0]#64*64*3
                            tmp_image[(2*i+1)*tmp.shape[0]>>2:(2*i+3)*tmp.shape[0]>>2,(2*j+1)*tmp.shape[1]>>2:(2*j+3)*tmp.shape[1]>>2] = tmp[tmp.shape[0]>>2:(3*tmp.shape[0]>>2),(tmp.shape[1]>>2):(3*tmp.shape[1]>>2)]
                    print("The middle part has been completed！")
                    
                    for j in tqdm(range(0,num_across)):
                        tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[0:sr_size,int((0.5*j)*sr_size):int((0.5*j+1)*sr_size)]]})[0]
                        tmp_image[0:int(0.75*tmp.shape[0]),int((j*0.5+0.25)*tmp.shape[1]):int((0.5*j+0.75)*tmp.shape[1])] = tmp[0:int(0.75*tmp.shape[0]),int(0.25*tmp.shape[1]):int(0.75*tmp.shape[1])]
                        tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[(-1)*sr_size:,int(0.5*j*sr_size):int((0.5*j+1)*sr_size)]]})[0]
                        tmp_image[int((-0.75)*tmp.shape[0]):,int((0.5*j+0.25)*tmp.shape[1]):int((0.5*j+0.75)*tmp.shape[1])] = tmp[int(0.25*tmp.shape[0]):,int(0.25*tmp.shape[1]):int(0.75*tmp.shape[1])]
                        
                    for i in tqdm(range(0,num_down)):
                        tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[int(0.5*i*sr_size):int((0.5*i+1)*sr_size),0:sr_size]]})[0]
                        tmp_image[int((0.5*i+0.25)*tmp.shape[0]):int((0.5*i+0.75)*tmp.shape[0]),0:int(0.75*tmp.shape[1])] = tmp[int(0.25*tmp.shape[0]):int(0.75*tmp.shape[0]),0:int(0.75*tmp.shape[1])]
                        tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[int(0.5*i*sr_size):int((0.5*i+1)*sr_size),(-1)*sr_size:]]})[0]
                        tmp_image[int((0.5*i+0.25)*tmp.shape[0]):int((0.5*i+0.75)*tmp.shape[0]),int((-0.75)*tmp.shape[1]):] = tmp[int(0.25*tmp.shape[0]):int(0.75*tmp.shape[0]),int(0.25*tmp.shape[1]):]
                    print("The edge part has been completed！")    
                    #[0,0]
                    tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[0:sr_size,0:sr_size]]})[0]
                    tmp_image[0:int(0.75*tmp.shape[0]),0:int(0.75*tmp.shape[1])] = tmp[0:int(0.75*tmp.shape[0]),0:int(0.75*tmp.shape[1])]          
                    
                    #[0,-1]
                    tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[0:sr_size,(-1)*sr_size:]]})[0]
                    tmp_image[0:int(0.75*tmp.shape[0]),int((-0.75)*tmp.shape[1]):] = tmp[0:int(0.75*tmp.shape[0]),int(0.25*tmp.shape[1]):]   
                    
                    #[-1,0]
                    tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[(-1)*sr_size:,0:sr_size]]})[0]
                    tmp_image[int((-0.75)*tmp.shape[0]):,0:int(0.75*tmp.shape[1])] = tmp[int(0.25*tmp.shape[0]):,0:int(0.75*tmp.shape[1])]          
                    
                    #[-1,-1]
                    tmp = sess.run(self.out,feed_dict={self.input:[input_tmp[(-1)*sr_size:,(-1)*sr_size:]]})[0]
                    tmp_image[int((-0.75)*tmp.shape[0]):,int((-0.75)*tmp.shape[1]):] = tmp[int(0.25*tmp.shape[0]):,int(0.25*tmp.shape[1]):]
                    
                    print("Image reconstruction has been completed!")
                    
                    output_tmp = tmp_image
                    
                output_img=img_as_ubyte(output_tmp/255)

                bic_img = cv2.resize(input_tmp,(input_img.shape[1],input_img.shape[0]),interpolation=cv2.INTER_CUBIC)
                bic_img=img_as_ubyte(bic_img/255)
                                   
                # print('-'*50)
                # print('img:',name)
                # print(input_tmp.shape)
                
                # plt.imshow(input_img)
                # plt.title('Grand true')
                # plt.show()
                
                # plt.imshow(input_tmp)
                # plt.title('input')
                # plt.show()
                # plt.imshow(output_img)
                # plt.title('output')
                # plt.show()
                
                eve_test[0]=num
                eve_test[1]=utils.com_psnr(input_img,output_img)
                eve_test[2]=utils.com_ypsnr(input_img,output_img)
                eve_test[3]=utils.com_ssim(input_img,output_img)
                eve_test[4]=utils.com_yssim(input_img,output_img)
                eve_test[5]=com_y_psnr_.get_psnr(com_y_psnr_.get_memary_img(input_img,output_img))
                eve_test[6]=com_y_psnr_.get_psnr(com_y_psnr_.get_memary_img(input_img,bic_img))

                test_mem=np.append(test_mem,eve_test,axis=1)
                print(eve_test)
                                    
                if img_is_GRAY:
                    output_img= cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
                imageio.imwrite(outputdir+'/'+img,output_img)
                print('-'*50)
                num=num+1
            print(test_mem)
            print('='*50)
            print(np.mean(test_mem, axis=1))
            np.save('test_mem',test_mem)
