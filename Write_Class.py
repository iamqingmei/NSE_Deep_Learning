class Write:
    """
    A class object, which will handle all the writing content to save into a txt file, after training and testing.
    """

    def __init__(self, write_content):
        """
        Initialize the Write object
        :param write_content: The content of Write object
        """
        self.write = write_content
        self.accuracies = []

    def add_content(self, write_content):
        """
        Add some content to current Write object
        :param write_content: The content to be added
        :return: None
        """
        self.write += write_content

    def add_accuracy(self, acc):
        """
        Add the accuracy to current Write object
        :param acc: The accuracy to be added
        :return: None
        """
        self.accuracies.append(acc)

    def save_write(self, folder_name):
        """
        Save all the write content into a txt file
        :param folder_name: The folder_name to save txt file
        :return: None
        """
        file_name = 'report_acc' + '_%0.2f' * len(self.accuracies) % tuple(self.accuracies) + '.txt'
        with open(folder_name + file_name, 'w') as f:
            f.truncate()
            f.write(self.write)
            f.close()

        print(self.write)
        print("Write saved")
