import pandas as pd
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # read data
    # shape of data_train and data_test (-1, 64, 64, 1)
    class_names_df = pd.read_csv('dataset/class_name.csv')
    class_names = class_names_df.to_numpy()
    data_train_df = pd.read_csv('dataset/data_train.csv')
    data_test_df = pd.read_csv('dataset/data_test.csv')
    label_train_df = pd.read_csv('dataset/label_train.csv')
    label_test_df = pd.read_csv('dataset/label_test.csv')
    print(f'data_train_df ')
    print(data_train_df.head())
    print(f'data_test_df ')
    print(data_test_df.head())
    print(f'label_test_df ')
    print(label_test_df.head())
    print(f'label_train_df ')
    print(label_train_df.head())
    data_train = data_train_df.to_numpy()
    data_train = data_train.reshape((-1, 64, 64, 1))
    label_train = label_train_df.to_numpy()

    data_test = data_test_df.to_numpy()
    data_test = data_test.reshape((-1, 64, 64, 1))
    label_test = label_test_df.to_numpy()

    def get_classname(label_number):
        return class_names[label_number]

    def show_images(img_data, img_label):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img_data[i], cmap='gray')
            plt.title(get_classname(img_label[i]))
            plt.axis("off")
        plt.show()

    show_images(data_train, label_train)
    show_images(data_test, label_test)