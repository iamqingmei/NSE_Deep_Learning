class Write:

    def __init__(self, write_content):
        self.write = write_content
        self.accuracies = []

    def add_content(self, write_content):
        self.write += write_content

    def add_accuracy(self, acc):
        self.accuracies.append(acc)

    def save_write(self, folder_name):

        file_name = 'report_acc' + '_%0.2f' * len(self.accuracies) % tuple(self.accuracies) + '.txt'
        with open(folder_name + file_name, 'w') as f:
            f.truncate()
            f.write(self.write)
            f.close()

        print(self.write)
        print("Write saved")

