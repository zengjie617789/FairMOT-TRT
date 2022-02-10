import cv2
import os
import glob

def draw_rectangle(data_path, img_id='00010.jpg', save_name='new_img.jpg', modified_name=None):
    dataset_name = os.path.split(data_path)[-1]
    img_path = os.path.join(data_path, img_id)
    label_path = img_path.replace('images', 'labels_with_ids').replace('.jpg', '.txt')
    if modified_name is not None:
        label_path = label_path.replace(dataset_name, modified_name)

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for label in labels:
            label_split = label.split(' ')
            cls, fid, x, y, w, h = label_split

            x0 = int(float(x) * img_w - float(w) * img_w / 2)
            y0 = int(float(y) * img_h - float(h) * img_h / 2)
            x1 = int(float(x) * img_w + float(w) * img_w / 2)
            y1 = int(float(y) * img_h + float(h) * img_h / 2)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.imwrite(save_name, img)


def rename_file(data_path):
    for i in sorted(os.listdir(data_path)):
        img_name = int(i.split('.')[0])
        new_name = '{:05d}'.format(img_name)+'.jpg'
        os.rename(os.path.join(data_path, i), os.path.join(data_path, new_name))
        print(img_name)


def main():
    # data_path = '/home/awifi/data/datasets/ServicesHallData/images/train06'  #for images_data
    data_path = '/home/awifi/data/datasets/ServicesHallData/images/train03'  #for images_data

    # data_path = '/home/awifi/data/ServicesHallData/images/train/output09/img1'  #for videos_data
    # data_path = '/home/awifi/data/datasets/crowdhuman/images/train/'
    # # label_path = '/home/awifi/data/datasets/ServicesHallData/labels_with_ids/train01/'
    # # data_path = '/home/awifi/data/ServicesHallData/images/test/output04/img1/'
    # # '/home/awifi/data/ServicesHallData/labels_with_ids/train/output01/img1/'
    draw_rectangle(data_path, img_id='00211.jpg', save_name='./test_outputs/img_test01.jpg')
    # rename_file(data_path)
    # draw_rectangle(data_path, 'img_bf.jpg', 'train03_bf' )


if __name__ == '__main__':
    main()