import pickle as pkl
import os
import json
import numpy as np
import argparse

is_vis=True
if is_vis:
    import cv2
    dir_imgs = 'data/test_images_2/'
    im_out='/home/dereyly/progs/tmp/tobacco/vis_cls/'
    os.makedirs(im_out,exist_ok=True)


class2name={}
th=1e-8
idx2cat=json.load(open('configs/idx2cat.json','r'))
max_ob=300
remap=pkl.load(open('configs/image_folder_idx.pkl','rb'))
invert={}
for key,val in remap.items():
    invert[val]=int(key)

data_out={}

def add_embedd(meta1,meta2,rebalance=False):
    for id, meta_boxes in meta1.items():
        meta_bb2=meta2[id]

        meta2[id][1]=(meta_bb2[1]*(1.1-meta2[id][2])+4*meta_boxes[1])/4

        meta2[id][2] = meta2[id][1].max()
        meta2[id][3] = meta2[id][1].argmax()

        zz=0
    return meta2

def recal_low_prob(meta,prob_th=0.8):
    meta_out=meta
    high_prob_list=[]
    for id, meta_boxes in meta.items():
        cls = meta_boxes[-1]
        prob = meta_boxes[-2]
        if prob>0.95:
            high_prob_list.append(cls)
    # high_prob_list+=black_list
    high_prob_list=np.array(high_prob_list)
    for id, meta_boxes in val.items():
        cls = meta_boxes[-1]
        prob = meta_boxes[-2]
        if prob<prob_th:
            new_res=meta_out[id][1].copy()
            new_res[high_prob_list]=0

            meta_out[id][-1] = new_res.argmax()
            meta_out[id][-2] = new_res.max()

    return meta_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_name', type=str)
    parser.add_argument('--pkl_embed',type = str)
    parser.add_argument('--json_out', type=str, help='path to save results', default='')
    args = parser.parse_args()
    data_pkl={}
    if len(args.json_out)==0:
        out_name = args.pkl_name[:-4] + '.json'
    else:
        out_name=args.json_out
    count_err=0
    with open(args.pkl_name,'rb') as f_in:
        data = pkl.load(f_in)
        data_embed=pkl.load(open(args.pkl_embed,'rb'))

        for key,val in data.items():
            is_err = False
            # if type(val)==dict and 'ensemble' in val:
            #     val=val['stage2']
            data_out[key]={}
            if is_vis:
                img = cv2.imread(dir_imgs + key)
            len_b=0
            val_embed=data_embed[key]
            val=add_embedd(val,val_embed)
            data_pkl[key]=val
            val=recal_low_prob(val)

            for id,meta_boxes in val.items():
                len_b+=1
                cls_embed=val_embed[id][-1]
                cls=meta_boxes[-1]
                prob=meta_boxes[-2]

                cls_in=cls
                cls=invert[cls]

                bb=meta_boxes[0]
                if not str(cls) in idx2cat:
                    print('no in idx')
                    continue
                cls_name = idx2cat[str(cls)]

                bb_out=bb[:4] #.astype(int).tolist()
                #bb_out = bb[:4].astype(int).tolist()
                if cls_name in data_out[key]:
                    data_out[key][cls_name].append(bb_out)
                else:
                    data_out[key][cls_name]=[bb_out]

                if is_vis:

                    if prob<0.7:
                        count_err+=1
                        cv2.rectangle(img, (bb_out[0], bb_out[1]), (bb_out[2], bb_out[3]), (255, 0, 0), 3)
                        #cv2.putText(img, '%.2f %d' % (prob,cls_name), (bb_out[0], bb_out[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0),2)
                        print(key,cls, prob)
                        is_err=True
                    #else:
                    cv2.putText(img, '%d' % (cls_name), (bb_out[0], bb_out[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if is_vis: # and is_err:
                cv2.imwrite(im_out + key, img)
            # print('num_boxes',len_b)
    # print(len(data_out),count_err)
    print('===> result file: ',out_name)
    json.dump(data_out,open(out_name,'w'),indent=4,sort_keys=True)
    pkl.dump(data_pkl,open(out_name+'.pkl','wb'))