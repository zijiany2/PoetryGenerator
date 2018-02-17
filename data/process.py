# -*- coding: utf-8 -*-
import codecs

data = codecs.open('duizhang.txt', 'r')
first_file = codecs.open('first.txt', 'w')
second_file = codecs.open('second.txt', 'w')
space_file = codecs.open('duizhang_space.txt', 'w')
data = data.readlines()
for d in data:
	tmp = d.split("，")
	if len(tmp) == 2:
		first = tmp[0]
		second = tmp[1][:-4]
		if len(first)==len(second):
			space_file.write(first + " " + second + "\n")
			first_file.write(first + "\n")
			second_file.write(second + "\n")
space_file.close()
first_file.close()
second_file.close()