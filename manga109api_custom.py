import manga109api


# This class uses the manga109api library to create a large dictionary with all relevant image data.
class Parser:
    def __init__(self, root_dir):
        """
        Initialize the Parser object with the root directory of the Manga109 dataset.
        """
        self.parser = manga109api.Parser(root_dir=root_dir)
        self.encoded_labels = {'face': 1, 'body': 2, 'text': 3, 'frame': 4}

    def load_all_images(self):

        img_path = []   # list of image paths
        width = []      # list of corresponding image widths
        height = []     # list of corresponding image heights
        bboxes = []     # list of corresponding bounding boxes
        labels = []     # list of corresponding labels
        book_id = []    # list of corresponding book titles

        for book in self.parser.books:
            book_annotation = self.parser.get_annotation(book=book)     # access book annotations
            page_annotations = book_annotation["page"]                  # access pages of book
            for page in page_annotations:                   # access each page annotation
                if self.validate_annotation(page):          # check to see if page has any bounding boxes

                    page_bboxes = []    # list of bounding boxes
                    page_labels = []    # list of corresponding bounding box labels

                    # add all relevant information to respective lists
                    for category in page.keys():
                        if category == '@index':
                            img_path.append(self.parser.img_path(book=book, index=page['@index']))
                        elif category == '@height':
                            height.append(page['@height'])
                        elif category == '@width':
                            width.append(page['@width'])
                        else:
                            for bbox in page[category]:
                                # [xmin, ymin, xmax, ymax] pascal_voc format
                                page_bboxes.append([bbox['@xmin'], bbox['@ymin'], bbox['@xmax'], bbox['@ymax']])
                                page_labels.append(self.encoded_labels[category])

                    labels.append(page_labels)
                    bboxes.append(page_bboxes)
                    book_id.append(book)

        # create final image dictionary
        img_dict = {
            "img_path": img_path,
            "width": width,
            "height": height,
            "book_id": book_id,
            "bboxes": bboxes,
            "labels": labels
        }

        return img_dict

    # Return True if a page annotation contains any of the specified annotation keys.
    def validate_annotation(self, page_annotation):
        for category in self.encoded_labels.keys():
            if page_annotation.get(category, []):  # check if the key exists and if its value is not an empty list
                return True
        return False
