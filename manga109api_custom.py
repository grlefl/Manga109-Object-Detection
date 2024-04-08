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
            for page in page_annotations:
                if self.validate_annotation(page):

                    page_bboxes = []
                    page_labels = []

                    for category in page.keys():
                        if category == '@index':
                            img_path.append(self.parser.img_path(book=book, index=page['@index']))
                        elif category == '@height':
                            height.append(page['@height'])
                        elif category == '@width':
                            width.append(page['@width'])
                        else:
                            for bbox in page[category]:
                                # [xmin, ymin, xmax, ymax] is correct format
                                page_bboxes.append([bbox['@xmin'], bbox['@ymin'], bbox['@xmax'], bbox['@ymax']])
                                page_labels.append(self.encoded_labels[category])

                    labels.append(page_labels)
                    bboxes.append(page_bboxes)
                    book_id.append(book)

        img_dict = {
            "img_path": img_path,
            "width": width,
            "height": height,
            "book_id": book_id,
            "bboxes": bboxes,
            "labels": labels
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
