GPU=0
ARCH=hg
DSET=coco
YEAR=2017
IMAGEPATH=~/dataset/coco/images  # set image path here
ANNOPATH=data/coco/coco_annotations_${YEAR}.json
STACK=1
BLOCK=1
FEAT=256
TRAINB=32
VALB=6
LR=1e-3
WORKERS=8


CHECKPOINT=./checkpoint/${DSET}/${ARCH}-s${STACK}-b${BLOCK}-batch${TRAINB}-lr${LR}


CUDA_VISIBLE_DEVICES=${GPU} python example/main.py \
--arch ${ARCH} \
--dataset ${DSET} \
--year ${YEAR} \
--image-path ${IMAGEPATH} \
--anno-path ${ANNOPATH} \
--stack ${STACK} \
--block ${BLOCK} \
--features ${FEAT} \
--checkpoint ${CHECKPOINT} \
--train-batch ${TRAINB} \
--test-batch ${VALB} \
--lr ${LR} \
--workers ${WORKERS} \
--target-weight
