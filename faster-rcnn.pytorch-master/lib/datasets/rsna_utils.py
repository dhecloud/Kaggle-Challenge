import pydicom as pdc
import cv2
import pandas as pd
import numpy as np
import os

data_dir = 'D:/all/'
def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the
    data into the following nested dictionary:

      parsed = {

        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': os.path.join(data_dir, 'stage_1_train_images/%s.dcm' % pid),
                'label': row['Target'],
                'boxes': [],
                'present': False}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))
            parsed[pid]['present'] = True


    return parsed

def DCM2RGB(im):
    im = np.stack([im] * 3, axis=2)
    return im

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present

    """
    # --- Open DICOM file
    d = pdc.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = DCM2RGB(im)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    print(im.shape)
    cv2.imshow("image", im)
    ch = cv2.waitKey(0)
    if ch == ord('q'):
        exit(0)

    im2 = d.pixel_array
    im2 = DCM2RGB(im2)
    im2 = cv2.resize(im2, (256, 256))
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im2 = overlay_box(im=im2, box=box, rgb=rgb, stroke=6)

    print(im2.shape)
    cv2.imshow("image", im2)
    ch = cv2.waitKey(0)
    if ch == ord('q'):
        exit(0)


def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    if im.shape[0] == 256:
        box = [int(b*0.25) for b in box]
    else:
        box = [int(b) for b in box]

    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

def inc_contrast(img, c, b):
    img = cv2.addWeighted(img, 1. + c/127., img, 0, b-c)
    return img

if __name__ == "__main__":
    # df = pd.read_csv(os.path.join(data_dir,'stage_1_train_labels.csv'))
    # parsed = parse_data(df)
    #
    # # len(parsed.keys())
    # for i in range(5256, len(parsed.keys())):
    #     print(df.iloc[i])
    #     patientId = df['patientId'][i]
    #     dcm_file = data_dir + '/stage_1_train_images/%s.dcm' % patientId
    #     dcm_data = pdc.read_file(dcm_file)
    #     assert(dcm_data.pixel_array.shape == (1024,1024))
    #     # draw(parsed[patientId])
    #     rgb = DCM2RGB(dcm_data.pixel_array)
    #     rgb = cv2.resize(rgb, (128, 128))
    #     jpg_file = data_dir + '/stage_1_train_images_resized_128/%s.jpg' % patientId
    #     cv2.imwrite(jpg_file, rgb)

    for (dirpath, dirnames, filenames) in os.walk(data_dir + '/stage_1_test_images/'):
        for file in filenames:
            if '.dcm' in file:
                dcm_data = pdc.read_file(data_dir + '/stage_1_test_images/' + file)
                print(dcm_data)
                assert(dcm_data.pixel_array.shape == (1024,1024))
                # draw(parsed[patientId])
                rgb = DCM2RGB(dcm_data.pixel_array)
                rgb = cv2.resize(rgb, (128, 128))
                jpg_file = data_dir + '/stage_1_test_images_resized_128/' + file.replace('.dcm','.jpg')
                cv2.imwrite(jpg_file, rgb)
