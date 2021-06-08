list =[]

with open("D:/PROJECTS/internship/3D_data/test_image_name.txt", "r") as o:
    contents = o.readlines()
    for i in contents:
        list.append(i.strip("\n"))
print(list[2])

