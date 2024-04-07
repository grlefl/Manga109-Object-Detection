import manga109api


class Parser:
    def __init__(self, root_dir):
        """
        Initialize the Parser object with the root directory of the Manga109 dataset.
        """
        self.parser = manga109api.Parser(root_dir=root_dir)
        self.encoded_labels = {'face': 1, 'body': 2, 'text': 3, 'frame': 4}

    def load_all_images(self):

        img_path = []
        width = []
        height = []
        bboxes = []
        labels = []
        book_id = []

        for book in self.parser.books:
            book_annotation = self.parser.get_annotation(book=book)
            page_annotations = book_annotation["page"]
            for annotation in page_annotations:
                if self.validate_annotation(annotation):

                    # img_path.append(annotation['path'])
                    width.append(annotation['@width'])
                    height.append(annotation['@height'])
                    book_id.append(book)

                    bboxes = []

                    for category in annotation.keys():
                        if category in self.encoded_labels.keys():
                            for bbox in category:
                                coco = [
                                    bbox['@xmin'],
                                    bbox['@ymin'],
                                    bbox['@xmax'] - bbox['@xmin'],  # width
                                    bbox['@ymax'] - bbox['@ymin']  # height
                                ]
                                bboxes.append(coco)  # [xmin, ymin, width, height]

                                labels.append(self.encoded_labels[category])

        img_dict = {
            "img_path": img_path,
            "width": width,
            "height": height,
            "bboxes": bboxes,
            "labels": labels,
            "book_id": book_id
        }

        return img_dict

    def validate_annotation(self, page_annotation):
        """
        Check if a page annotation contains any of the specified annotation keys.

        Returns:
        - bool: True if the page annotation contains any of the specified categories, False otherwise.
        """
        for category in self.encoded_labels.keys():
            if page_annotation.get(category, []):  # check if the key exists and if its value is not an empty list
                return True
        return False