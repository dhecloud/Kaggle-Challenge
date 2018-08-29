import pydicom as pdc
import matplotlib.pyplot as plt
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
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present

    """
    # --- Open DICOM file
    d = pdc.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    plt.imshow(im, cmap=plt.cm.bone)
    plt.axis('off')
    plt.show()

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
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
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(data_dir,'stage_1_train_labels.csv'))
    parsed = parse_data(df)

    for i in range(46, len(parsed.keys())):
        print(df.iloc[i])
        patientId = df['patientId'][46]
        dcm_file = data_dir + '/stage_1_train_images/%s.dcm' % patientId
        dcm_data = pdc.read_file(dcm_file)
        print(parsed[patientId])
        assert(dcm_data.pixel_array.shape == (1024,1024))
        print(len(parsed[patientId]['boxes']))
        # assert(len(parsed[patientId]['boxes']) == 2 or len(parsed[patientId]['boxes']) == 0)
        draw(parsed[patientId])
        # plt.imshow(dcm_data.pixel_array, cmap=plt.cm.bone)
        # plt.axis('off')
        # plt.show()
