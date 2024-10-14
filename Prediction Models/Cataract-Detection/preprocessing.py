import cv2
import os

def image_processing(raw_data,data_path,height,width):
    class_labels=[]
    category_count=0
    for i in os.walk(raw_data):
        if len(i[2])>0:
            counter=0
            images=i[2]
            class_name=i[0].strip('\\')
            print(class_name)
            path=os.path.join(data_path,class_labels[category_count])
            for image in images:
                im=cv2.imread(class_name+'\\'+image)
                im=cv2.resize(im,(height,width))
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path,str(counter)+'.jpg'),im)
                counter+=1
            category_count+=1
        else:
            number_of_classes=len(i[1])
            print(number_of_classes,i[1])
            class_labels=i[1][:]

if __name__=='__main__':
    height = 100
    width = 100
    raw_data = 'rawdata'
    data_path = 'data'
    if not os.path.exists(data_path):
        image_processing(raw_data, data_path, height, width)


