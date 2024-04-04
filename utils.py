
# from the github
def load_all_images(parser, classes):
    """
    Using the parser from manga109api_custom, for every book load all the image paths and annotations and store them in a list of lists.
    Args:
      Parser (object)
    Returns:
      images (List[List[str, Dict]])
      author_labels (List[int])
    """
    images = []
    author_labels = []
    for book in parser.books:
      annotation, author = parser.get_annotation(book=book)
      for i in range(0, len(annotation["page"])):
        temp=[]
        if check_annotation_validity(annotation, i, classes):
          path=parser.img_path(book=book, index=i)
          temp.append(path)
          temp.append(annotation["page"][i])
          images.append(temp)
          author_labels.append(author)
    return images, author_labels

# from the github
def check_annotation_validity(annotation, index, classes):
    """
    This function check the presence of at least one annotated box in the image.
    """
    count = 0
    for cl in classes[1:]:
        if len(annotation["page"][index][str(cl)])==0:
            count += 1
    if count == len(classes[1:]):
        return False
    return True