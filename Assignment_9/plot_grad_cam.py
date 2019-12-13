def plot_grad_cam(model, last_conv_layer, image):
    class_idx = np.argmax(model.predict(image)[0])
    class_output = model.output[:,class_idx]
    
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads,axis=(0,1,2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    
    pooled_grads, last_conv_layer_output = iterate([image])
    for i in range(pooled_grads.shape[0]):
        last_conv_layer_output[:,:,i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    import cv2
    heatmap = cv2.resize(heatmap, (image.shape[1],image.shape[2]))
    fig,ax = plt.subplots(1,2)
    
    ax[0].imshow(image[0])
    
    ax[1].imshow(image[0],alpha=0.9)
    ax[1].imshow(heatmap,alpha=0.6)
