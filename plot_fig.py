def vis_detections(im, dets = None, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    if len(im.shape) == 3:
        im = im[:, :, (2, 1, 0)]
    if len(im.shape) == 3:
        plt.imshow(im)
    else:
        plt.imshow(im, extent=[0, 1, 0, 1])    
    if dets != None:
      for i in xrange(np.minimum(10, dets.shape[0])):
          
          bbox = dets[i, :4]
          score = dets[i, -1]
          if score > thresh:
              plt.gca().add_patch(
                  plt.Rectangle((bbox[0], bbox[1]),
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1], fill=False,
                                edgecolor='g', linewidth=3)
                )
            #plt.title('{}  {:.3f}'.format(class_name, score))
    plt.show()
    time.sleep(5)
    plt.close()
    
def save_imagesc(im , detection = None, name = 'lilac'):
  import cv2
  mini = im.min()
  maxi = im.max()
  #print mini,maxi
  im = im - mini
  im = im/(maxi-mini)
  im = (im*255.0)
  
  if detection != None:
    for  i in xrange(detection.shape[0]):
      det = detection[i,:]
      det = det.astype('int32')
      im[max(det[1],0):min(det[3],im.shape[0]),max(0,det[0]-3):min(im.shape[1],det[0]+3)] = 255.0 * ((i+0.0)/detection.shape[0])
      im[max(det[1],0):min(det[3],im.shape[0]),max(0,det[2]-3):min(im.shape[1],det[2]+3)] = 255.0 * ((i+0.0)/detection.shape[0])
      im[max(det[1]-3,0):min(det[1]+3,im.shape[0]),max(0,det[0]):min(im.shape[1],det[2])] = 255.0 * ((i+0.0)/detection.shape[0])
      im[max(det[3]-3,0):min(det[3]+3,im.shape[0]),max(0,det[0]):min(im.shape[1],det[2])] = 255.0 * ((i+0.0)/detection.shape[0])
    
  #print im.shape
  im = im.reshape((im.shape[0],im.shape[1],1))
  #im = cv2.resize(im,(128,128))
  #print im.shape
  cv2.imwrite(name +'.jpg', im)
  
