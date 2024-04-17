import os
import shutil


# # mkdir
# allFileList = os.listdir("./tw_food_101/train/")
# for file in allFileList:
#     print("./tw_food_101/valid/" + file)
#     os.mkdir("./tw_food_101/valid/" + file)


# # Train 1/4 move to valid (dataset download from kaggle)
# root_dir, cur_dir, _ = next(os.walk("./tw_food_101/train/"))
# for sub_dir in cur_dir:
#     for _, _, files in os.walk(root_dir + sub_dir):
#         [count_files, count_move] = [len(files), len(files) // 4]
#         print(count_files, count_move, root_dir + sub_dir)
#         for index, file in enumerate(files):
#             if index < count_move:
#                 shutil.move(
#                     root_dir + sub_dir + "/" + file,
#                     "./tw_food_101/valid/" + sub_dir + "/" + file,
#                 )


# # Train 1/4 move to valid (dataset download from 106368015AlvinYang)
# with open("./tw_food_101/validation.txt", "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         path = line.split("\n")[0]
#         print(path)
#         shutil.move("./tw_food_101/train/" + path, "./tw_food_101/valid/" + path)

# folderList = os.listdir("./tw_food_101/valid/")
# for folder in folderList:
#     print(
#         "./tw_food_101/valid/"
#         + folder
#         + "--------------------------------------------------------------------------------------------------------"
#     )
#     fileList = os.listdir("./tw_food_101/valid/" + folder)
#     for index, file in enumerate(fileList):
#         endswith = file.split(".")[-1]
#         oldFileName = "./tw_food_101/valid/" + folder + "/" + file
#         newFileName = "./tw_food_101/valid/" + folder + "/" + f"{index+1}." + endswith
#         print(f"{index+1}")
#         print(oldFileName)
#         print(newFileName)
#         # shutil.copy(oldFileName, "./tw_food_101/dasdsadasdas/" + file)
#         os.rename(oldFileName, newFileName)
