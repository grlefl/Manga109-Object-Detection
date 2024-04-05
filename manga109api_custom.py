import manga109api


class Parser:
    def __init__(self, root_dir):
        """
        Initialize the Parser object with the root directory of the Manga109 dataset.
        """
        self.parser = manga109api.Parser(root_dir=root_dir)

    def load_all_images(self):
        """
        Load all annotated images from the Manga109 dataset along with their book labels.

        Returns:
        - tuple: (images, book_labels) - lists containing image dictionaries and corresponding book labels.
        """
        images = []
        book_labels = []
        for book_title in self.parser.books:
            book_annotation = self.parser.get_annotation(book=book_title)
            pages = book_annotation["page"]
            for page_annotation in pages:
                if validate_annotations(page_annotation):
                    page_index = page_annotation["@index"]
                    img_path = self.parser.img_path(book=book_title, index=page_index)
                    images.append({"img_path": img_path, "page_annotation": page_annotation})
                    book_labels.append(book_title)
        return images, book_labels


def validate_annotations(page_annotation):
    """
    Check if a page annotation contains any of the specified annotation keys.

    Returns:
    - bool: True if the page annotation contains any of the specified annotation keys, False otherwise.
    """
    annotation_keys = {"body", "face", "frame", "text"}
    return annotation_keys.intersection(page_annotation.keys())
