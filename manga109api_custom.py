import manga109api


class Parser:
    def __init__(self, root_dir):
        self.parser = manga109api.Parser(root_dir=root_dir)

    def load_all_images(self):
        images = []
        book_labels = []
        for book_title in self.parser.books:
            book_annotation = self.parser.get_annotation(book=book_title)
            pages = book_annotation["page"]
            for page_annotation in pages:
                page_index = page_annotation["@index"]
                img_path = self.parser.img_path(book=book_title, index=page_index)
                images.append({"img_path": img_path, "page_annotation": page_annotation})
                book_labels.append(book_title)
        return images, book_labels
