from PIL import ImageDraw, Image


def draw_rectangle(image, x0, y0, x1, y1, annotation_type):
    assert annotation_type in ["body", "face", "frame", "text"]

    color = {"body": "#258039", "face": "#f5be41",
             "frame": "#31a9b8", "text": "#cf3721"}[annotation_type]
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0, y0, x1, y1), outline=color, width=10)


def draw_annotations(image, page_annotation):
    # image = Image.open(img_path)

    for annotation_type in ["body", "face", "frame", "text"]:
        rois = page_annotation[annotation_type]
        # regions of interest
        for roi in rois:
            draw_rectangle(image, int(roi["@xmin"]), int(roi["@ymin"]), int(roi["@xmax"]), int(roi["@ymax"]),
                           annotation_type)
    return image
