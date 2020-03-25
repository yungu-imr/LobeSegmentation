from keras.layers import Input, BatchNormalization, MaxPool3D, Conv3D, UpSampling3D, Concatenate, Activation, add, multiply, Lambda
from keras.models import Model
from keras.optimizers import Adam
from dice import *
import keras.backend as K


def unet_residual(input_shape):
	"""只包括相对位置和肺裂隙感知的网络模型
	"""
    num_classes = 6
    inputs = Input(input_shape)
    dists = Input((input_shape[0], input_shape[1], input_shape[2], 3))

    '''downsample'''
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(inputs)
    batc1 = BatchNormalization(axis=-1)(conv1)
    acti1 = Activation('relu')(batc1)
    conv2 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti1)
    batc2 = BatchNormalization(axis=-1)(conv2)
    acti2 = Activation('relu')(batc2)
    maxp1 = MaxPool3D(2)(acti2)


    conv3 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1)(conv3)
    batc3 = add([batc3, maxp1])
    acti3 = Activation('relu')(batc3)
    conv4 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(acti3)
    batc4 = BatchNormalization(axis=-1)(conv4)
    acti4 = Activation('relu')(batc4)
    maxp2 = MaxPool3D(2)(acti4)


    conv5 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1)(conv5)
    batc5 = add([batc5, maxp2])
    acti5 = Activation('relu')(batc5)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(acti5)
    batc6 = BatchNormalization(axis=-1)(conv6)
    acti6 = Activation('relu')(batc6)
    maxp3 = MaxPool3D(2)(acti6)

    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1)(conv7)
    batc7 = add([batc7, maxp3])
    acti7 = Activation('relu')(batc7)
    conv42 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(acti7)
    batc42 = BatchNormalization(axis=-1)(conv42)
    acti42 = Activation('relu')(batc42)
    maxp4 = MaxPool3D(2)(acti42)

    conv51 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(maxp4)
    batc51 = BatchNormalization(axis=-1)(conv51)
    batc51 = add([batc51, maxp4])
    acti51 = Activation('relu')(batc51)    
    conv8 = Conv3D(256, 3, padding='same', kernel_initializer='he_normal')(acti51)
    batc8 = BatchNormalization(axis=-1)(conv8)
    acti8 = Activation('relu')(batc8)


    '''upsample'''
    upsa = UpSampling3D(2)(acti8)
    merg = Concatenate(axis=-1)([conv42, upsa])
    conv61 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(merg)
    batc61 = BatchNormalization(axis=-1)(conv61)
    acti61 = Activation('relu')(batc61)
    conv62 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(acti61)
    batc62 = BatchNormalization(axis=-1)(conv62)
    batc62 = add([batc62, acti61])
    acti62 = Activation('relu')(batc62)

    upsa1 = UpSampling3D(2)(acti62)
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([conv6, upsa1])
    conv9 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merg1)
    batc9 = BatchNormalization(axis=-1)(conv9)
    acti9 = Activation('relu')(batc9)
    conv10 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(acti9)
    batc10 = BatchNormalization(axis=-1)(conv10)
    batc10 = add([batc10, acti9])
    acti10 = Activation('relu')(batc10)

    upsa2 = UpSampling3D(2)(acti10)
    merg2 = Concatenate(axis=-1)([conv4, upsa2])
    conv11 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    conv12 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(acti11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    batc12 = add([batc12, acti11])
    acti12 = Activation('relu')(batc12)

    upsa3 = UpSampling3D(2)(acti12)
    merg3 = Concatenate(axis=-1)([conv2, upsa3, dists])
    #merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv13 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    conv14 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti13)
    convol = Conv3D(num_classes, 1, activation='softmax')(conv14)
    conedge = Conv3D(2, 1, activation='softmax')(conv14)

    model = Model(inputs=[inputs, dists], outputs=[convol, conedge])
    # model = Model(inputs=inputs, outputs=convol)

    return model

def unet_residual_nopos(input_shape):
	"""只包括肺裂隙感知的网络模型
	"""
    num_classes = 6
    inputs = Input(input_shape)
    dists = Input((input_shape[0], input_shape[1], input_shape[2], 3))

    '''downsample'''
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(inputs)
    batc1 = BatchNormalization(axis=-1)(conv1)
    acti1 = Activation('relu')(batc1)
    conv2 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti1)
    batc2 = BatchNormalization(axis=-1)(conv2)
    acti2 = Activation('relu')(batc2)
    maxp1 = MaxPool3D(2)(acti2)


    conv3 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1)(conv3)
    batc3 = add([batc3, maxp1])
    acti3 = Activation('relu')(batc3)
    conv4 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(acti3)
    batc4 = BatchNormalization(axis=-1)(conv4)
    acti4 = Activation('relu')(batc4)
    maxp2 = MaxPool3D(2)(acti4)


    conv5 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1)(conv5)
    batc5 = add([batc5, maxp2])
    acti5 = Activation('relu')(batc5)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(acti5)
    batc6 = BatchNormalization(axis=-1)(conv6)
    acti6 = Activation('relu')(batc6)
    maxp3 = MaxPool3D(2)(acti6)

    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1)(conv7)
    batc7 = add([batc7, maxp3])
    acti7 = Activation('relu')(batc7)
    conv42 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(acti7)
    batc42 = BatchNormalization(axis=-1)(conv42)
    acti42 = Activation('relu')(batc42)
    maxp4 = MaxPool3D(2)(acti42)

    conv51 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(maxp4)
    batc51 = BatchNormalization(axis=-1)(conv51)
    batc51 = add([batc51, maxp4])
    acti51 = Activation('relu')(batc51)    
    conv8 = Conv3D(256, 3, padding='same', kernel_initializer='he_normal')(acti51)
    batc8 = BatchNormalization(axis=-1)(conv8)
    acti8 = Activation('relu')(batc8)


    '''upsample'''
    upsa = UpSampling3D(2)(acti8)
    merg = Concatenate(axis=-1)([conv42, upsa])
    conv61 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(merg)
    batc61 = BatchNormalization(axis=-1)(conv61)
    acti61 = Activation('relu')(batc61)
    conv62 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(acti61)
    batc62 = BatchNormalization(axis=-1)(conv62)
    batc62 = add([batc62, acti61])
    acti62 = Activation('relu')(batc62)

    upsa1 = UpSampling3D(2)(acti62)
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([conv6, upsa1])
    conv9 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merg1)
    batc9 = BatchNormalization(axis=-1)(conv9)
    acti9 = Activation('relu')(batc9)
    conv10 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(acti9)
    batc10 = BatchNormalization(axis=-1)(conv10)
    batc10 = add([batc10, acti9])
    acti10 = Activation('relu')(batc10)

    upsa2 = UpSampling3D(2)(acti10)
    merg2 = Concatenate(axis=-1)([conv4, upsa2])
    conv11 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    conv12 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(acti11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    batc12 = add([batc12, acti11])
    acti12 = Activation('relu')(batc12)

    upsa3 = UpSampling3D(2)(acti12)
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    #merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv13 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    conv14 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti13)
    convol = Conv3D(num_classes, 1, activation='softmax')(conv14)
    conedge = Conv3D(2, 1, activation='softmax')(conv14)

    model = Model(inputs=[inputs, dists], outputs=[convol, conedge])
    # model = Model(inputs=inputs, outputs=convol)

    return model


def unet_residual_fissureAtt(input_shape):
	"""包括相对位置、肺裂隙感知和肺裂隙增强的网络模型
	"""
    num_classes = 6
    inputs = Input(input_shape)
    dists = Input((input_shape[0], input_shape[1], input_shape[2], 3))
    fissure_att = Input(input_shape)

    '''fissure attetion'''
    input_fissure = Concatenate(axis=-1)([inputs, fissure_att])
    conv1_att = Conv3D(16, 3, padding='same', kernel_initializer='he_normal', name='conv1_att')(input_fissure)
    conv1_att = BatchNormalization(axis=-1, name='conv1_att_bn')(conv1_att)
    conv1_att = Activation('relu')(conv1_att)
    conv2_att = Conv3D(16, 3, padding='same', kernel_initializer='he_normal', name='conv2_att')(conv1_att)
    conv2_att = BatchNormalization(axis=-1, name='conv2_att_bn')(conv2_att)
    conv2_att = Activation('relu')(conv2_att)
    maxp_att = MaxPool3D(2)(conv2_att)
    conv3_att = Conv3D(64, 3, padding='same', kernel_initializer='he_normal', name='conv3_att')(maxp_att)
    conv3_att = BatchNormalization(axis=-1, name='conv3_att_bn')(conv3_att)
    conv3_att = Activation('relu')(conv3_att)
    conv4_att = Conv3D(16, 3, padding='same', kernel_initializer='he_normal', name='conv4_att')(conv3_att)
    conv4_att = BatchNormalization(axis=-1, name='conv4_att_bn')(conv4_att)
    conv4_att = Activation('relu')(conv4_att)
    conv5_att = Conv3D(1, 3, padding='same', kernel_initializer='he_normal', activation='sigmoid',name='conv5_att')(conv4_att)
    sig_att = Lambda(lambda x: K.tile(x, (1,1,1,1,16)))(conv5_att)
    

    '''downsample'''
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(inputs)
    batc1 = BatchNormalization(axis=-1)(conv1)
    acti1 = Activation('relu')(batc1)
    conv2 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti1)
    batc2 = BatchNormalization(axis=-1)(conv2)
    acti2 = Activation('relu')(batc2)
    maxp1 = MaxPool3D(2)(acti2)


    conv3 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1)(conv3)
    batc3 = add([batc3, maxp1])
    acti3 = Activation('relu')(batc3)
    att_acti3 = multiply([sig_att, acti3])
    # att_acti3 = acti3
    conv4 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(att_acti3)
    batc4 = BatchNormalization(axis=-1)(conv4)
    acti4 = Activation('relu')(batc4)
    maxp2 = MaxPool3D(2)(acti4)


    conv5 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1)(conv5)
    batc5 = add([batc5, maxp2])
    acti5 = Activation('relu')(batc5)
    # att_acti5 = multiply([sig_att, acti5]) 
    att_acti5 = acti5
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(att_acti5)
    batc6 = BatchNormalization(axis=-1)(conv6)
    acti6 = Activation('relu')(batc6)
    maxp3 = MaxPool3D(2)(acti6)

    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1)(conv7)
    batc7 = add([batc7, maxp3])
    acti7 = Activation('relu')(batc7)
    conv42 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(acti7)
    batc42 = BatchNormalization(axis=-1)(conv42)
    acti42 = Activation('relu')(batc42)
    maxp4 = MaxPool3D(2)(acti42)

    conv51 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(maxp4)
    batc51 = BatchNormalization(axis=-1)(conv51)
    batc51 = add([batc51, maxp4])
    acti51 = Activation('relu')(batc51)
    # att_acti51 = multiply([sig_att, acti51])    
    conv8 = Conv3D(256, 3, padding='same', kernel_initializer='he_normal')(acti51)
    batc8 = BatchNormalization(axis=-1)(conv8)
    acti8 = Activation('relu')(batc8)


    '''upsample'''
    upsa = UpSampling3D(2)(acti8)
    merg = Concatenate(axis=-1)([conv42, upsa])
    conv61 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(merg)
    batc61 = BatchNormalization(axis=-1)(conv61)
    acti61 = Activation('relu')(batc61)
    conv62 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(acti61)
    batc62 = BatchNormalization(axis=-1)(conv62)
    batc62 = add([batc62, acti61])
    acti62 = Activation('relu')(batc62)

    upsa1 = UpSampling3D(2)(acti62)
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([conv6, upsa1])
    conv9 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merg1)
    batc9 = BatchNormalization(axis=-1)(conv9)
    acti9 = Activation('relu')(batc9)
    conv10 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(acti9)
    batc10 = BatchNormalization(axis=-1)(conv10)
    batc10 = add([batc10, acti9])
    acti10 = Activation('relu')(batc10)

    upsa2 = UpSampling3D(2)(acti10)
    merg2 = Concatenate(axis=-1)([conv4, upsa2])
    conv11 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    conv12 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(acti11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    batc12 = add([batc12, acti11])
    acti12 = Activation('relu')(batc12)

    upsa3 = UpSampling3D(2)(acti12)
    merg3 = Concatenate(axis=-1)([conv2, upsa3, dists])
    #merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv13 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    conv14 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti13)
    convol = Conv3D(num_classes, 1, activation='softmax')(conv14)
    conedge = Conv3D(2, 1, activation='softmax')(conv14)

    model = Model(inputs=[inputs, dists, fissure_att], outputs=[convol, conedge])
    # model = Model(inputs=inputs, outputs=convol)

    return model