import os
top_dir = '.'
f = open('files.txt','a+')
d = open('directories.txt','a+')
def os_list_dir(top_dir,d,f):  
  for file in os.listdir(top_dir):  
    file_path = os.path.abspath(os.path.join(top_dir, file))  
    if os.path.isfile(file_path):
      if file_path[-3:] != 'mat' and file_path[-3:] != '.so' and file_path[-3:] != 'rc3' and file_path[-3:] != 'jpg' and file_path[-3:] != 'xml' and file_path[-3:] != 'del' and file_path[-3:] != 'png' and file_path[-4:] != '.txt' and file_path[-4:] != '.pyc' and file_path[-2:] != '.o' and file_path[-2:] != '.d' and file_path[-3:] != 'pkl': 
        print >>f,'%s'%file_path
    elif os.path.isdir(file_path):  
      print >>d,'%s'%file_path  
      os_list_dir(file_path,d,f)
os_list_dir(top_dir,d,f)
f.close()