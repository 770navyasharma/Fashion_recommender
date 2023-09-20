import os
import shutil
import random

# Aapke data folders ke paths
main_data_folder = 'men_women_classification//data'  # Isko actual path se replace karein

# Train, validation, aur test folders banayein (agar nahi hain)
train_men_folder = 'men_women_classification//train_data//women'
validation_men_folder = 'men_women_classification//validation//women'
test_men_folder = 'men_women_classification//test//women'


# Men folder ke path
men_folder = 'men_women_classification//data//women'

# Men images ki list banaayein
men_images = os.listdir(men_folder)

# Men images ko shuffle karein
random.shuffle(men_images)

# Men images ki total count nikalein
total_men_images = len(men_images)

# Train, validation, aur test mein kitni images rakhni hain, woh calculate karein
train_men_count = int(0.8 * total_men_images)
validation_men_count = int(0.1 * total_men_images)
test_men_count = int(0.1 * total_men_images)

# Ab men images ko train, validation, aur test folders mein move karein
for i in range(train_men_count):
    src_path = os.path.join(men_folder, men_images[i])
    dst_path = os.path.join(train_men_folder, men_images[i])
    shutil.move(src_path, dst_path)

for i in range(train_men_count, train_men_count + validation_men_count):
    src_path = os.path.join(men_folder, men_images[i])
    dst_path = os.path.join(validation_men_folder, men_images[i])
    shutil.move(src_path, dst_path)

for i in range(train_men_count + validation_men_count, train_men_count + validation_men_count + test_men_count):
    src_path = os.path.join(men_folder, men_images[i])
    dst_path = os.path.join(test_men_folder, men_images[i])
    shutil.move(src_path, dst_path)
