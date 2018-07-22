#CUDA_VISIBLE_DEVICES=2 python2 tools/infer_simple.py \
    #--cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    #--output-dir DensePoseData/infer_out/ \
    #--image-ext png \
    #--wts https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
        #test_imgs/0005.png

CUDA_VISIBLE_DEVICES=2 python2 tools/infer_densepose.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ \
    --image-ext png \
    --wts https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
        test_imgs/0005.png
