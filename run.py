import os

for run in range(1, 17):
    print (run)
    os.system('python3 app.py --input_uri ../camloc/oct22/'+str(run)+'.mp4 --mot --gui -l results.txt -o output.mp4')
    os.system('mv pixel-locs.txt camera_'+str(run)+'.txt')
