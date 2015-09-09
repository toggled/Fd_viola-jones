import os
for filename in os.listdir('.'):
    #print filename
    if 'gif' not in filename and 'py' not in filename:
        print filename
        os.rename(filename,filename+'.gif')